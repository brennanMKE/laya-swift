// Ported from laya-go internal/tokenizer/tokenizer.go (added tokens).
//
// Tokens are compared as UTF-8 bytes. Swift's String == uses canonical
// equivalence, so it would match "[SEP]" against text that Python and the
// Rust tokenizers treat as different.

struct AddedToken: Sendable {
    var content: [UInt8]
    var id: Int32
    var lstrip: Bool
    var rstrip: Bool
}

/// Finds added tokens in text, leftmost-longest.
struct AddedTokenMatcher: Sendable {
    /// Candidates by first byte, longest first.
    private var byFirst: [UInt8: [AddedToken]] = [:]

    init(_ tokens: [AddedToken]) {
        for token in tokens {
            byFirst[token.content[0], default: []].append(token)
        }
        for (first, candidates) in byFirst {
            // A stable sort, so equal lengths keep file order as in laya-go.
            byFirst[first] = candidates.enumerated()
                .sorted { a, b in
                    a.element.content.count != b.element.content.count
                        ? a.element.content.count > b.element.content.count
                        : a.offset < b.offset
                }
                .map(\.element)
        }
    }

    /// Calls `emit` for each piece of `bytes[range]` in order: an added token
    /// with its id, or a run of other text with nil. lstrip and rstrip tokens
    /// absorb the whitespace next to them, as in the Rust implementation.
    func split(_ bytes: [UInt8], _ range: Range<Int>, emit: (Range<Int>, Int32?) -> Void) {
        if byFirst.isEmpty {
            if !range.isEmpty { emit(range, nil) }
            return
        }
        var start = range.lowerBound // start of pending plain text
        var i = start
        while i < range.upperBound {
            guard let token = longest(in: bytes, at: i, end: range.upperBound) else {
                i += 1
                continue
            }
            var s = i, e = i + token.content.count
            if token.lstrip {
                while s > start {
                    let (scalar, size) = decodeLastScalar(bytes, before: s)
                    if !scalar.isPatternSpace { break }
                    s -= size
                }
            }
            if token.rstrip {
                while e < range.upperBound {
                    let (scalar, size) = decodeScalar(bytes, at: e)
                    if !scalar.isPatternSpace { break }
                    e += size
                }
            }
            if s > start { emit(start..<s, nil) }
            emit(s..<e, token.id)
            start = e
            i = e
        }
        if start < range.upperBound { emit(start..<range.upperBound, nil) }
    }

    private func longest(in bytes: [UInt8], at i: Int, end: Int) -> AddedToken? {
        guard let candidates = byFirst[bytes[i]] else { return nil }
        return candidates.first { token in
            token.content.count <= end - i && bytes[i..<(i + token.content.count)].elementsEqual(token.content)
        }
    }
}
