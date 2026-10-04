import Foundation
import LayaCore
import LayaHub
import LayaMetal

/// Laya in Swift. A placeholder until phases 1–4 of docs/swift-port.md land:
/// it reports where the model would run, but can't answer questions yet.
public actor LayaModel {
    public struct Info: Sendable {
        /// "metal" when the GPU engine can run here, otherwise nil.
        public var engine: String?
        /// Why the GPU engine can't run, when it can't.
        public var engineFallback: String?
        public var device: String?
        public var checkpoint: String
    }

    public init() {}

    /// Checks this device without loading weights.
    public func info() -> Info {
        let checkpoint = "\(LayaCheckpoint.repo)@\(LayaCheckpoint.revision.prefix(7))"
        do {
            let kernels = try MetalKernels.load()
            return Info(engine: "metal", engineFallback: nil, device: kernels.deviceName, checkpoint: checkpoint)
        } catch {
            return Info(engine: nil, engineFallback: String(describing: error), device: nil, checkpoint: checkpoint)
        }
    }
}
