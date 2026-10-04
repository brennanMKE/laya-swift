import Foundation
import Testing
@testable import LayaCore

/// The fixtures copied from laya-go by scripts/sync-from-laya-go.sh load,
/// and their checkpoint metadata decodes.
@Suite struct GoldenFixtureTests {
    static let golden = Bundle.module.resourceURL!.appendingPathComponent("golden")

    @Test func metaDecodes() throws {
        let meta = try CheckpointConfig(contentsOf: Self.golden.appendingPathComponent("meta.json"))
        #expect(meta.maxLen == 512)
        #expect(meta.headMaxLen == 192)
        #expect(meta.temperature.count == QuestionType.allCases.count)
        #expect(meta.temperatureByOptions["noul:2"] != nil)
    }

    @Test func fixturesPresent() throws {
        for name in ["games.jsonl", "edge.jsonl", "calibration.jsonl", "tokenizer.jsonl"] {
            let url = Self.golden.appendingPathComponent(name)
            let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int ?? 0
            #expect(size > 0, "\(name) is missing or empty")
        }
    }
}
