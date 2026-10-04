import Foundation

/// The parts of a checkpoint's rl_agent_config.json the text side needs.
public struct CheckpointConfig: Decodable, Sendable, Equatable {
    public var maxLen: Int
    public var headMaxLen: Int
    /// Temperatures by question type id (choice, score, noul).
    public var temperature: [Double]
    /// Temperatures by bucket, such as "choice:3-5" or "noul:2".
    public var temperatureByOptions: [String: Double]

    enum CodingKeys: String, CodingKey {
        case maxLen = "max_len"
        case headMaxLen = "head_max_len"
        case temperature
        case temperatureByOptions = "temperature_by_options"
    }

    public init(contentsOf url: URL) throws {
        self = try JSONDecoder().decode(CheckpointConfig.self, from: Data(contentsOf: url))
    }
}
