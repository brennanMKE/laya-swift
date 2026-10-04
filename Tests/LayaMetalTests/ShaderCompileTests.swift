import Testing
@testable import LayaMetal

/// laya-go's shaders compile with precise math on this Mac's GPU.
@Suite struct ShaderCompileTests {
    @Test func shadersCompile() throws {
        let kernels: MetalKernels
        do {
            kernels = try MetalKernels.load()
        } catch MetalEngineError.noDevice, MetalEngineError.unsupportedGPU {
            return // nothing to test here (e.g. a VM or an Intel Mac)
        }
        #expect(!kernels.functionNames.isEmpty)
        #expect(kernels.functionNames.contains { $0.contains("gemm") })
    }
}
