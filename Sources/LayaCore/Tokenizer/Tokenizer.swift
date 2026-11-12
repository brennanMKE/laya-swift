// Ported from laya-go internal/tokenizer/tokenizer.go.

import Foundation
import os

/// The Hugging Face tokenizers pipeline that Laya's ModernBERT checkpoint
/// ships in tokenizer/tokenizer.json:
///
///  1. added tokens are split out first: tokens with normalized=false (the
///     special tokens) on the raw text, then the others on the NFC-normalized
///     remainder, both leftmost-longest;
///  2. the NFC normalizer;
///  3. the ByteLevel pre-tokenizer with the GPT-2 split pattern;
///  4. byte-level BPE with merge ranks;
///  5. optionally, the TemplateProcessing post-processor ([CLS] A [SEP]).
///
/// It supports exactly the configuration types that tokenizer.json uses (NFC,
/// ByteLevel, BPE, TemplateProcessing) and rejects others when loading, so an
/// unsupported tokenizer fails loudly rather than tokenizing differently.
public final class Tokenizer: Sendable {
    /// Vocabulary and added tokens, keyed by the token's UTF-8 bytes.
    private let vocab: [[UInt8]: Int32]
    private let added: [[UInt8]: Int32]
    /// The vocabulary id of each byte's byte-level character, if it has one.
    private let byteIDs: [Int32?]
    /// Merges keyed by the pair of ids (see `pairKey`).
    private let merges: [UInt64: Merge]

    /// Added tokens with normalized=false, matched on raw text.
    private let raw: AddedTokenMatcher
    /// Added tokens with normalized=true, matched on NFC text.
    private let normalized: AddedTokenMatcher

    private let single: [TemplatePiece]?
    private let pair: [TemplatePiece]?

    /// Pre-tokenized word (raw bytes) to BPE ids.
    private let cache = OSAllocatedUnfairLock<[[UInt8]: [Int32]]>(initialState: [:])

    private struct Merge: Sendable {
        var rank: Int32
        var id: Int32
    }

    /// One element of a post-processor template: fixed special-token ids, or
    /// a slot for sequence A or B.
    private enum TemplatePiece: Sendable {
        case special([Int32])
        case sequence(String)
    }

    /// Bounds the per-word BPE cache. Game prompts reuse a small vocabulary,
    /// so a modest cache gets nearly every word.
    private static let maxCache = 1 << 16

    /// Reads a tokenizer.json file.
    public convenience init(contentsOf url: URL) throws {
        do {
            try self.init(data: Data(contentsOf: url))
        } catch let error as TokenizerError {
            throw TokenizerError.invalid("\(url.path): \(error.message)")
        }
    }

    /// Parses a tokenizer.json document.
    ///
    /// JSONSerialization keeps NSDictionary keys distinct by their code
    /// units. Swift's Dictionary<String, …> would merge keys that are only
    /// canonically equivalent, which Python treats as different tokens.
    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? NSDictionary else {
            throw TokenizerError.invalid("not a JSON object")
        }
        if let normalizer = root["normalizer"] as? NSDictionary, normalizer["type"] as? String != "NFC" {
            throw TokenizerError.invalid("unsupported normalizer \(normalizer["type"] ?? "nil")")
        }
        guard let pre = root["pre_tokenizer"] as? NSDictionary, pre["type"] as? String == "ByteLevel",
              pre["use_regex"] as? Bool == true, pre["add_prefix_space"] as? Bool != true else {
            throw TokenizerError.invalid("unsupported pre-tokenizer (want ByteLevel, use_regex, no prefix space)")
        }
        guard let model = root["model"] as? NSDictionary, model["type"] as? String == "BPE",
              (model["dropout"] as? Double ?? 0) == 0, model["byte_fallback"] as? Bool != true,
              model["ignore_merges"] as? Bool != true,
              (model["continuing_subword_prefix"] as? String ?? "").isEmpty,
              (model["end_of_word_suffix"] as? String ?? "").isEmpty,
              let vocabJSON = model["vocab"] as? NSDictionary,
              let mergesJSON = model["merges"] as? NSArray else {
            throw TokenizerError.invalid("unsupported model: want plain byte-level BPE")
        }

        var vocab = [[UInt8]: Int32](minimumCapacity: vocabJSON.count)
        for case let (token as String, id as Int) in vocabJSON {
            vocab[Array(token.utf8)] = Int32(id)
        }
        self.vocab = vocab
        // Bytes that never occur in UTF-8 (0xC0, 0xC1, 0xF5–0xFF) may be
        // missing from the vocab; like the Rust BPE without an unk token,
        // bpe drops them.
        byteIDs = byteToScalar.map { vocab[Array(String($0).utf8)] }

        var merges = [UInt64: Merge](minimumCapacity: mergesJSON.count)
        for (rank, entry) in mergesJSON.enumerated() {
            let (a, b) = try Self.parseMerge(entry, rank: rank)
            guard let ia = vocab[a], let ib = vocab[b], let id = vocab[a + b] else {
                throw TokenizerError.invalid("merge \(rank) uses tokens missing from the vocab")
            }
            let key = Self.pairKey(ia, ib)
            if merges[key] == nil {
                merges[key] = Merge(rank: Int32(rank), id: id)
            }
        }
        self.merges = merges

        var added = [[UInt8]: Int32]()
        var raws: [AddedToken] = [], norms: [AddedToken] = []
        for case let entry as NSDictionary in root["added_tokens"] as? NSArray ?? [] {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else {
                throw TokenizerError.invalid("added token without content or id")
            }
            if entry["single_word"] as? Bool == true {
                throw TokenizerError.invalid("added token \(content.debugDescription): single_word is not supported")
            }
            if content.isEmpty { continue }
            added[Array(content.utf8)] = Int32(id)
            var token = AddedToken(content: Array(content.utf8), id: Int32(id),
                                   lstrip: entry["lstrip"] as? Bool == true,
                                   rstrip: entry["rstrip"] as? Bool == true)
            if entry["normalized"] as? Bool == true && entry["special"] as? Bool != true {
                token.content = Array(content.precomposedStringWithCanonicalMapping.utf8)
                norms.append(token)
            } else {
                raws.append(token)
            }
        }
        self.added = added
        raw = AddedTokenMatcher(raws)
        normalized = AddedTokenMatcher(norms)

        if let post = root["post_processor"] as? NSDictionary {
            guard post["type"] as? String == "TemplateProcessing" else {
                throw TokenizerError.invalid("unsupported post-processor \(post["type"] ?? "nil")")
            }
            let specials = post["special_tokens"] as? NSDictionary ?? [:]
            single = try Self.parseTemplate(post["single"], specials)
            pair = try Self.parseTemplate(post["pair"], specials)
        } else {
            single = nil
            pair = nil
        }
    }

    /// Accepts both merge spellings: "a b" and ["a", "b"].
    private static func parseMerge(_ entry: Any, rank: Int) throws -> ([UInt8], [UInt8]) {
        if let parts = entry as? NSArray {
            guard parts.count == 2, let a = parts[0] as? String, let b = parts[1] as? String else {
                throw TokenizerError.invalid("merge \(rank): want 2 parts")
            }
            return (Array(a.utf8), Array(b.utf8))
        }
        guard let s = entry as? String else {
            throw TokenizerError.invalid("merge \(rank): not a string or a pair")
        }
        let bytes = Array(s.utf8)
        guard let space = bytes.firstIndex(of: UInt8(ascii: " ")) else {
            throw TokenizerError.invalid("merge \(rank): no space in \(s.debugDescription)")
        }
        return (Array(bytes[..<space]), Array(bytes[(space + 1)...]))
    }

    private static func parseTemplate(_ json: Any?, _ specials: NSDictionary) throws -> [TemplatePiece] {
        var pieces: [TemplatePiece] = []
        for case let entry as NSDictionary in json as? NSArray ?? [] {
            if let special = entry["SpecialToken"] as? NSDictionary, let name = special["id"] as? String {
                guard let token = specials[name] as? NSDictionary, let ids = token["ids"] as? [Int] else {
                    throw TokenizerError.invalid("post-processor: unknown special token \(name.debugDescription)")
                }
                pieces.append(.special(ids.map(Int32.init)))
            } else if let sequence = entry["Sequence"] as? NSDictionary, let name = sequence["id"] as? String {
                pieces.append(.sequence(name))
            } else {
                throw TokenizerError.invalid("post-processor: unknown template entry \(entry)")
            }
        }
        return pieces
    }

    @inline(__always)
    private static func pairKey(_ a: Int32, _ b: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))
    }

    /// The id of a vocabulary or added token, compared byte for byte.
    public func tokenID(_ token: String) -> Int32? {
        let bytes = Array(token.utf8)
        return added[bytes] ?? vocab[bytes]
    }

    /// Tokenizes text without adding special tokens, like
    /// `tokenizer(text, add_special_tokens=False)["input_ids"]` in Python.
    public func encode(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        append(text, to: &ids)
        return ids
    }

    /// Appends the ids of text (no special tokens added) to `ids`.
    public func append(_ text: String, to ids: inout [Int32]) {
        let text = Array(text.utf8)
        // Pass 1: special (non-normalized) added tokens on the raw text.
        raw.split(text, text.indices) { range, id in
            if let id {
                ids.append(id)
                return
            }
            // Pass 2: normalize the rest, then split on normalized added tokens.
            let piece = Array(String(decoding: text[range], as: UTF8.self).precomposedStringWithCanonicalMapping.utf8)
            normalized.split(piece, piece.indices) { range, id in
                if let id {
                    ids.append(id)
                    return
                }
                preTokenize(piece, range) { word in
                    appendWord(Array(piece[word]), to: &ids)
                }
            }
        }
    }

    /// Tokenizes text and applies the post-processor's single template
    /// ([CLS] text [SEP] for ModernBERT), like `tokenizer(text)` in Python.
    public func encodeWithSpecialTokens(_ text: String) -> [Int32] {
        applyTemplate(single, encode(text), [])
    }

    /// Tokenizes a pair of texts and applies the pair template
    /// ([CLS] a [SEP] b [SEP] for ModernBERT).
    public func encodePair(_ a: String, _ b: String) -> [Int32] {
        applyTemplate(pair, encode(a), encode(b))
    }

    private func applyTemplate(_ template: [TemplatePiece]?, _ a: [Int32], _ b: [Int32]) -> [Int32] {
        guard let template else { return a + b }
        var out: [Int32] = []
        for piece in template {
            switch piece {
            case .special(let ids): out += ids
            case .sequence("A"): out += a
            case .sequence("B"): out += b
            case .sequence: break
            }
        }
        return out
    }

    /// Appends the BPE ids of one pre-tokenized word (raw bytes).
    private func appendWord(_ word: [UInt8], to ids: inout [Int32]) {
        if let cached = cache.withLock({ $0[word] }) {
            ids += cached
            return
        }
        let wordIDs = bpe(word)
        cache.withLock { cache in
            if cache.count >= Self.maxCache { cache.removeAll(keepingCapacity: true) }
            cache[word] = wordIDs
        }
        ids += wordIDs
    }

    /// Byte-level BPE over the bytes of word: each byte starts as its own
    /// symbol, and the adjacent pair with the lowest merge rank (leftmost on
    /// ties) is merged until no pair has a rank. This is the order the Rust
    /// implementation's priority queue produces.
    private func bpe(_ word: [UInt8]) -> [Int32] {
        var symbols = word.compactMap { byteIDs[Int($0)] }
        while symbols.count > 1 {
            var best: Merge?
            var at = -1
            for i in 0..<(symbols.count - 1) {
                if let m = merges[Self.pairKey(symbols[i], symbols[i + 1])], best == nil || m.rank < best!.rank {
                    best = m
                    at = i
                }
            }
            guard let best else { break }
            symbols[at] = best.id
            symbols.remove(at: at + 1)
        }
        return symbols
    }
}

public struct TokenizerError: Error, CustomStringConvertible {
    public var message: String

    static func invalid(_ message: String) -> TokenizerError { TokenizerError(message: message) }

    public var description: String { "tokenizer: \(message)" }
}
