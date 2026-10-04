import Foundation
import LayaCore
import Metal

/// Why the Metal engine can't run on this device.
public enum MetalEngineError: Error, Equatable, CustomStringConvertible {
    case noDevice
    /// The kernels use simdgroup matrices, which need the Apple7 GPU family (A14 or M1 and later).
    case unsupportedGPU(String)
    case shaderCompile(String)

    public var description: String {
        switch self {
        case .noDevice: "metal: no Metal device"
        case .unsupportedGPU(let name): "metal: \(name) lacks simdgroup matrix support (needs Apple7 family or newer)"
        case .shaderCompile(let message): "metal: shaders failed to compile: \(message)"
        }
    }
}

/// The Metal device and compiled kernels. The forward pass comes in phase 3
/// of docs/swift-port.md; for now this checks support and compiles the
/// shaders exactly as laya-go does.
public struct MetalKernels: Sendable {
    public let deviceName: String
    /// The kernel functions in the compiled library.
    public let functionNames: [String]
    public let compileSeconds: Double

    public static func load() throws -> MetalKernels {
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalEngineError.noDevice }
        guard device.supportsFamily(.apple7) else { throw MetalEngineError.unsupportedGPU(device.name) }
        let options = MTLCompileOptions()
        // Precise math, as laya-go compiles them: fast math changes GELU, exp and LayerNorm.
        if #available(iOS 18, macOS 15, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        let start = Date()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: ShaderSource.laya, options: options)
        } catch {
            throw MetalEngineError.shaderCompile(error.localizedDescription)
        }
        return MetalKernels(
            deviceName: device.name,
            functionNames: library.functionNames.sorted(),
            compileSeconds: Date().timeIntervalSince(start)
        )
    }
}
