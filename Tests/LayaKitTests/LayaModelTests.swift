import Testing
@testable import LayaKit

@Suite struct LayaModelTests {
    @Test func infoReportsEngineOrReason() async {
        let info = await LayaModel().info()
        #expect((info.engine == nil) != (info.engineFallback == nil))
        #expect(info.checkpoint.hasPrefix("convaiinnovations/laya@"))
    }
}
