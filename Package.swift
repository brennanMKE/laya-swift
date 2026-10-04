// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LayaKit", targets: ["LayaKit"]),
        .library(name: "LayaCore", targets: ["LayaCore"]),
        .executable(name: "laya-swift", targets: ["LayaCLI"]),
    ],
    targets: [
        .target(name: "LayaCore"),
        .target(name: "LayaMetal", dependencies: ["LayaCore"]),
        .target(name: "LayaHub", dependencies: ["LayaCore"]),
        .target(name: "LayaKit", dependencies: ["LayaCore", "LayaMetal", "LayaHub"]),
        .executableTarget(name: "LayaCLI", dependencies: ["LayaKit"]),
        .testTarget(
            name: "LayaCoreTests",
            dependencies: ["LayaCore"],
            resources: [.copy("Resources/golden")]
        ),
        .testTarget(name: "LayaMetalTests", dependencies: ["LayaMetal"]),
        .testTarget(name: "LayaKitTests", dependencies: ["LayaKit"]),
    ],
    swiftLanguageModes: [.v6]
)
