import Foundation
import LayaHub
import LayaKit
import LayaMetal

// laya-swift: development tool. For now, `laya-swift info` checks the
// GPU, compiles the shaders and looks for the checkpoint.

let args = CommandLine.arguments.dropFirst()
guard args.first == "info" || args.isEmpty else {
    print("usage: laya-swift info")
    exit(2)
}

do {
    let kernels = try MetalKernels.load()
    print("GPU: \(kernels.deviceName)")
    print("Shaders: \(kernels.functionNames.count) kernels compiled in \(Int(kernels.compileSeconds * 1000)) ms")
} catch {
    print("GPU: unavailable (\(error))")
}
if let snapshot = LayaCheckpoint.cachedSnapshot(in: LayaCheckpoint.defaultHubCache) {
    print("Checkpoint: \(snapshot.path)")
} else {
    print("Checkpoint: \(LayaCheckpoint.repo)@\(LayaCheckpoint.revision.prefix(7)) not in the Hugging Face cache")
}
