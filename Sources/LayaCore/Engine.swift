/// The boundary between the text side and an engine, mirroring laya-go's
/// engine.Engine (see laya-go's CONTRACT.md).

/// The three question types, with the ids Laya's model uses.
public enum QuestionType: Int, Sendable, CaseIterable {
    case choice = 0
    case score = 1
    case noul = 2
}

/// Packed sequences for one forward pass.
public struct LayaBatch: Sendable {
    /// Token ids of every sequence, back to back.
    public var tokens: [Int32]
    /// Where each sequence starts in `tokens`, plus a final entry equal to `tokens.count`.
    public var starts: [Int]
    /// Marker positions per sequence, relative to the sequence's start.
    public var markers: [[Int]]
    /// Question type per sequence.
    public var questionTypes: [QuestionType]
    /// Whether to compute the act head's logits.
    public var wantAct: Bool

    public init(tokens: [Int32], starts: [Int], markers: [[Int]], questionTypes: [QuestionType], wantAct: Bool = true) {
        self.tokens = tokens
        self.starts = starts
        self.markers = markers
        self.questionTypes = questionTypes
        self.wantAct = wantAct
    }

    public var sequenceCount: Int { starts.count - 1 }
}

/// Raw outputs of one forward pass: one logit per marker, and the act head's logits.
public struct LayaEngineOutput: Sendable {
    public var logits: [[Float]]
    public var actLogits: [[Float]]

    public init(logits: [[Float]], actLogits: [[Float]]) {
        self.logits = logits
        self.actLogits = actLogits
    }
}

/// Runs the network. Temperatures, softmax and rounding stay on the text side.
public protocol LayaEngine: Sendable {
    /// A short name such as "metal fp16 weights (Apple M4 Pro)".
    var name: String { get }
    func forward(_ batch: LayaBatch) async throws -> LayaEngineOutput
}
