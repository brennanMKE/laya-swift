import Foundation
import Testing
@testable import LayaCore

@Suite(.enabled(if: TestCheckpoint.tokenizerURL != nil, "no checkpoint"))
struct TokenizerTests {
    /// Loaded once for the suite; a checkpoint that fails to load fails the tests.
    static let loaded = Result { try Tokenizer(contentsOf: TestCheckpoint.tokenizerURL!) }

    var tokenizer: Tokenizer { get throws { try Self.loaded.get() } }

    @Test func examples() throws {
        let tok = try tokenizer
        #expect(tok.encode("a  b").contains(50276), "two spaces are an added token")
        #expect(tok.encode("x [SEP] y").contains(50282), "[SEP] in text is the separator")
        #expect(tok.encodeWithSpecialTokens("hi") == [50281] + tok.encode("hi") + [50282])
        let pair = tok.encodePair("a", "b")
        #expect(pair.count == 5 && pair[0] == 50281 && pair[2] == 50282 && pair[4] == 50282)
        #expect(tok.tokenID("[MASK]") != nil)
        #expect(tok.encode("") == [])
    }

    /// Every string in golden/tokenizer.jsonl, which make_golden.py tokenized
    /// with the Python tokenizer.
    @Test func golden() throws {
        let tok = try tokenizer
        struct Record: Decodable {
            var text: String
            var ids: [Int32]
        }
        let url = Bundle.module.resourceURL!.appendingPathComponent("golden/tokenizer.jsonl")
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        var bad = 0
        for line in lines {
            let record = try JSONDecoder().decode(Record.self, from: Data(line.utf8))
            let got = tok.encode(record.text)
            if got != record.ids {
                bad += 1
                if bad <= 20 {
                    Issue.record("\(record.text.debugDescription):\n got  \(got)\n want \(record.ids)")
                }
            }
        }
        #expect(lines.count == 6511)
        #expect(bad == 0, "\(lines.count - bad) of \(lines.count) strings match")
    }
}
