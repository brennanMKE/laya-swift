# Laya in Swift: design for a Swift package

This is a design for running [Laya](https://huggingface.co/convaiinnovations/laya) inference in
Swift, as a Swift package that SwiftUI apps on iOS, iPadOS and macOS can add. laya-go is the
reference implementation: the Swift port reuses its Metal shaders (`internal/metal/shaders.metal`)
and its golden fixtures (`testdata/golden`), and it follows laya-go's split between the text side
and the engine ([CONTRACT.md](../CONTRACT.md)). This page lives in laya-go for that reason.

**Status:** proposal. Nothing here is built. The name **LayaKit** is a working name, not a
decision; so are every type and target name below. Facts about laya-go come from this repository
at `00f1442` (version 0.2.1); facts about Laya from laya 0.3.6 and checkpoint
`convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`, the versions laya-go is tested
against. Measurements are laya-go's, taken on an M4 Pro (16-core GPU, 64 GB). Nothing has been run
on an iPhone or iPad. Anything marked **Unverified** was not checked, and anything marked
**Estimate** was not measured.

## Summary

- **Port laya-go to Swift, file for file; don't wrap a runtime.** The text side (tokenizer, prompt
  building, Python-compatible JSON, calibration) is about 1,500 lines of Go with no dependencies
  beyond `golang.org/x/text`. The Metal engine is about 1,300 lines of Go host code plus 452 lines
  of Metal Shading Language. In Swift the host code gets simpler: Metal is a native API, with no
  purego, no Objective-C runtime calls by selector, and no locked OS threads.
- **Run the same Metal kernels, unchanged.** The shaders are already measured: on the M4 Pro, 16 ms
  for one state, 54 ms for 6 states and 130 ms for a 14-state Tetris batch, every logit within
  1.4e-4 of PyTorch fp32 and every choice the same. A Swift host that dispatches the same kernels
  with the same arguments should give the same logits as laya-go on the same Mac (**Unverified**,
  but cheap to check, and the first parity test to write).
- **Target Apple7 GPUs and newer: A14 / M1 and later.** The kernels need `simdgroup_matrix`, which
  is Apple7 and later. That covers iPhone 12 and later, iPads with A14, A15, A17 Pro or an M chip,
  and every Apple silicon Mac. Older devices and the iOS Simulator (which offers only Apple2-class
  Metal) have no GPU engine; see [CPU fallback](#cpu-fallback-and-the-simulator).
- **Minimum OS: iOS 17 and macOS 14, by choice.** The Metal features alone would allow iOS 14 and
  macOS 11. iOS 17 and macOS 14 are chosen for Observation, `DispatchSerialQueue` as an actor
  executor and Swift 6 concurrency. This is far below the iOS / macOS 27 that
  [system-one-foundation-models](#related-work-system-one-foundation-models) needs for Foundation
  Models.
- **Weights download at run time, never in the app bundle.** 842.6 MB (804 MiB) of fp16
  safetensors from Hugging Face at a pinned revision, checked against their sha256, into
  Application Support and excluded from backup. The resident cost on the GPU is about 0.95 GB.
- **Prove parity with laya-go's golden fixtures.** Token ids identical; raw logits within 1e-3;
  4-decimal answers equal in at least 99.5% of fields; every argmax the same. The fixtures are
  20 MB and run on device as a test resource bundle.
- **Expect an iPhone to be 3–6 times slower than the M4 Pro** (**Estimate**, from GPU throughput and
  memory bandwidth ratios): roughly 45–120 ms for one state and 150–400 ms for 6 states,
  depending on the chip. Usable for interactive decisions; tight for realtime games without the
  answer cache.
- **Complement system-one-foundation-models, don't compete with it.** That package bridges System
  One models into Apple's Foundation Models framework, and its on-device path is Core ML in fp16
  with a tokenizer that doesn't reproduce Hugging Face's ids. A small adapter conforming to its
  `SystemOneBackend` protocol would give its users laya-go-level accuracy and batching on the GPU.

## Why

Laya answers typed questions about a state in one forward pass. On a phone that makes it a good
on-device decision model: private, offline, deterministic, and with no generation loop. What
exists today doesn't serve SwiftUI apps well:

- **laya-go** runs on macOS, but not on iOS. Its Metal engine calls Metal through purego and the
  Objective-C runtime, which is built for `CGO_ENABLED=0` Go binaries. Shipping a Go runtime inside
  an iOS app is possible (see [Alternatives](#alternatives-considered)) but awkward.
- **The Python package** needs PyTorch and doesn't run on iOS at all.
- **system-one-foundation-models' `LayaOnDevice`** runs on device through Core ML, but its
  tokenizer, calibration and fp16 precision mean its answers won't match Laya's (see
  [Related work](#related-work-system-one-foundation-models)). It also requires OS 27.

A Swift port gives apps a native API (`async`, actors, `Sendable` value types), runs the
already-proven kernels, and keeps laya-go's parity bar.

## Goals and non-goals

Goals:

- Answer Laya's `choice`, `score` and `noul` questions with the same answers as laya 0.3.6 within
  the tolerances below, for the English checkpoint `convaiinnovations/laya`.
- A Swift package with a small public API mirroring laya-go: load with progress, `predict`,
  `predictMany`, `info`.
- iOS 17+, iPadOS 17+ and macOS 14+ on Apple7-or-newer GPUs, with a clear "unsupported" result
  elsewhere and an optional CPU fallback on macOS.
- Weights downloaded at run time from the Hugging Face Hub at a pinned revision, resumable and
  verified, stored the way iOS expects for re-downloadable data.
- One forward pass per call, with all of a call's sequences packed into one command buffer, as
  laya-go does.
- A SwiftUI sample app that shows download, loading, asking and lifecycle handling.

Non-goals:

- Training, fine-tuning or fitting temperatures.
- Other architectures or arbitrary Hugging Face models.
- Bit-exact equality with PyTorch. The tolerances define "the same".
- The multilingual and typed-decisions checkpoints in the first release (they can follow, as in
  laya-go's plan).
- The HTTP server. Apps call the library; a Mac app that wants a server can run `laya-server`.
- watchOS and tvOS. visionOS is likely easy (Apple silicon GPU) but untested and not a goal yet.

## What Laya computes

The model, the sequence format and calibration are described precisely in System 1 Arcade's
[Laya in Go design](https://github.com/brennanMKE/System1Arcade/blob/main/docs/laya-go-port.md#what-laya-computes)
and implemented in laya-go. In short:

- **One sequence per (state, question):**
  `[CLS] "<type> question: <instructions>" [SEP] [MASK] " opt0" [MASK] " opt1" … [SEP] <state> [SEP]`,
  at most `max_len` (512) tokens, options squeezed to fit `head_max_len` (192). Each option's
  `[MASK]` position is a marker. Ported in `internal/prompt/prompt.go` (157 lines).
- **The network:** a 28-layer ModernBERT-large encoder (hidden 1024, 16 heads, GeGLU MLP, RoPE with
  θ 160000 on the global layers 0, 3, … 27 and θ 10000 with a ±64 sliding window on the rest), a
  final LayerNorm, a per-type embedding added to every token, two pre-norm
  `nn.TransformerEncoderLayer` head layers with **ReLU** (not GELU), then a scorer (LayerNorm, Linear, GELU,
  Linear(1)) at each marker and an act head over `[h[CLS], top1, top1 − top2, entropy / log k, k / 255]`.
- **Calibration:** a temperature per `"<type>:<size>"` bucket clamped to [0.5, 5.0], a float32
  softmax, then the answer per type (`choice` argmax with `confidence = 1 − H(p)/log k`, `score`
  `Σ i·pᵢ`, `noul` `p[1]` with `max(p, 1 − p)`), rounded to 4 decimals the way Python's
  `round(x, 4)` does.

The engine boundary is the same as laya-go's: the engine returns raw scorer logits at each marker
and the two raw act-head logits; temperatures, softmax, confidence and rounding stay in the text
side.

## Public API (sketch)

Swift names follow laya-go's where they translate. Everything a caller holds is a `Sendable`
value type; the model is an actor.

```swift
import LayaKit // proposed name

// Loading -----------------------------------------------------------------

public struct LayaOptions: Sendable {
    public var repo = "convaiinnovations/laya"
    public var revision = LayaCheckpoint.pinnedRevision   // "1c5edc17…"
    public var directory: URL?             // a local checkpoint directory instead of the Hub
    public var cacheDirectory: URL?        // default: Application Support/LayaKit/hub
    public var token: String?              // Hugging Face token, sent only to huggingface.co
    public var offline = false             // never download
    public var allowsCellular = false      // 843 MB: off by default
    public var engine: LayaEngineChoice = .auto   // .auto, .metal, .cpu
    public init() {}
}

public enum LoadProgress: Sendable {
    case downloading(file: String, done: Int64, total: Int64)
    case verifying(file: String)
    case loadingWeights(done: Int, total: Int)
    case compilingKernels
}

public actor LayaModel {
    public static func load(_ options: LayaOptions = .init(),
                            progress: (@Sendable (LoadProgress) -> Void)? = nil) async throws -> LayaModel
    public nonisolated let info: LayaInfo
    public func predict(_ state: LayaState, _ questions: [NamedQuestion]) async throws -> LayaResult
    public func predictMany(_ prompts: [LayaPrompt]) async throws -> [LayaResult]
    public func encode(_ prompts: [LayaPrompt]) throws -> [LayaSequence]   // token ids and markers, for tools
    public func close()                    // frees the GPU buffers; later calls throw .closed
}

public struct LayaInfo: Sendable {
    public let checkpoint: URL
    public let revision: String
    public let maxLen: Int, headMaxLen: Int
    public let temperatureByType: [QuestionKind: Double]
    public let temperatureByBucket: [String: Double]
    public let clamped: [String]           // e.g. "choice:11+=0.1006"
    public let engine: String              // "metal fp16 weights (Apple A17 Pro GPU)"
    public let engineFallback: String?     // why .auto didn't use Metal
}

// Questions ---------------------------------------------------------------

public enum QuestionKind: UInt8, Sendable, Hashable { case choice = 0, score = 1, noul = 2 }

public struct Question: Sendable, Hashable {
    public static func choice(_ instructions: String, _ options: [ChoiceOption]) -> Question
    public static func choice(_ instructions: String, sortedFrom: [String: String]) -> Question // laya-go's ChoiceMap
    public static func score(_ instructions: String, levels: [String]) -> Question
    public static func noul(_ instructions: String, whenTrue: String? = nil, whenFalse: String? = nil) -> Question
}

public struct ChoiceOption: Sendable, Hashable {
    public var name: String
    public var description: String?        // nil or "" means "no description", as in Python
}

public struct NamedQuestion: Sendable { public var name: String; public var question: Question }

public struct LayaState: Sendable {
    public static func text(_ s: String) -> LayaState
    public static func json(_ data: Data) throws -> LayaState   // serialized as Python's json.dumps would
}

public struct LayaPrompt: Sendable {
    public var key: String                 // batch key
    public var state: LayaState
    public var questions: [NamedQuestion]  // in order
}

// Answers -----------------------------------------------------------------

public struct LayaResult: Sendable {
    public let key: String
    public let names: [String]             // question order
    public let answers: [String: Answer]
    public let inputTokens: Int
}

public struct Answer: Sendable {
    public let kind: QuestionKind
    public let choice: String?             // choice
    public let score: Double?              // score: Σ i·pᵢ
    public let noul: Double?               // noul: P(true)
    public let probabilities: [Probability]   // option order; empty for noul
    public let legend: [String]            // score levels
    public let confidence: Double
    public let actProbability: Double
}

public struct Probability: Sendable, Hashable { public let label: String; public let p: Double }

public enum LayaError: Error, Sendable {
    case optionsTooLong(key: String, question: String, headMaxLen: Int)
    case noOptions(key: String, question: String)
    case duplicateOption(key: String, question: String, name: String)
    case unsupportedGPU(String)            // no Apple7 GPU (A12/A13, Intel, Simulator)
    case insufficientMemory(available: Int, needed: Int)
    case notInForeground                   // iOS: GPU work isn't allowed in the background
    case closed
    case incompatibleCheckpoint(String)
    case download(file: String, underlying: String)
    case checksumMismatch(file: String)
}
```

Notes:

- **Option order is part of the prompt**, as in laya-go. `Question.choice(_:_:)` takes an array;
  `sortedFrom:` sorts by name, which is what System 1 Arcade sends and what
  system-one-foundation-models' sequence builder does with its `[String: String]` criteria.
- **Numbers are full precision**; `Answer.rounded()` gives Python's 4-decimal values. An optional
  `LayaWire` type can encode and decode laya-go's `/predict` JSON for apps that talk to the same
  contract; it is not needed for the library.
- **Answer cache.** `LayaAnswerCache` ports `answercache/answercache.go`: a bounded LRU keyed by
  `json.dumps([state, type, instructions, criteria], ensure_ascii=False)`, the same key laya-go
  and System 1 Arcade use, with the same hit, miss, call and skip counts. It wraps a model:
  `cache.predictMany(prompts, using: model)` asks the model only for misses, still as one batch.
  10,000 answers is a few MB.
- **Engine protocol.** The boundary is laya-go's `engine.Engine`:

  ```swift
  public struct LayaBatch: Sendable {
      public var tokens: [Int32]           // all sequences, packed
      public var starts: [Int32]           // n + 1 offsets
      public var markers: [[Int32]]        // per sequence, relative to its start
      public var qtypes: [UInt8]
  }
  public struct LayaEngineOutput: Sendable {
      public var logits: [[Float]]         // raw scorer logits, marker order
      public var actLogits: [SIMD2<Float>] // [act, escalate]
  }
  public protocol LayaEngine: AnyObject, Sendable {
      var name: String { get }
      func forward(_ batch: LayaBatch) async throws -> LayaEngineOutput
      func close() async
  }
  ```

  `LayaModel.init(checkpoint:engine:)` takes any engine, like `laya.New`, so tests can pass a stub
  that returns fixture logits, and the sample app's SwiftUI previews need no weights.

### Concurrency model

- **`LayaModel` is an actor.** It owns the tokenizer, the prompt builder, the temperatures and the
  engine. Text work (tokenizing, building sequences, calibrating) is microseconds to a millisecond
  per call and runs on the actor.
- **The Metal engine is an actor on its own serial executor**, a `DispatchSerialQueue` (usable as
  a `SerialExecutor` from iOS 17 / macOS 14). It encodes one command buffer per `forward`, commits
  it and blocks that queue in `waitUntilCompleted`, as laya-go does. Blocking a dedicated queue for
  16–400 ms is fine; blocking Swift's cooperative thread pool for that long is not.
- **Why not `await` the GPU with `addCompletedHandler` instead?** Because actors are reentrant: a
  second `forward` would enter while the first one's command buffer is in flight and overwrite the
  shared activation buffers (laya-go's `workspace`). Either the engine serializes with a blocking
  wait on its own queue (proposed), or it keeps an explicit async queue of calls. One workspace
  and one pass at a time is what laya-go measured, and the GPU is saturated by one pass anyway.
- **`Sendable` everywhere at the boundary.** `MTLBuffer`, `MTLDevice` and pipeline states never
  leave the engine actor. Results are copied out of the shared buffers before `forward` returns,
  as laya-go copies `ws.logit`.
- **Cancellation.** `predictMany` checks `Task.checkCancellation()` before tokenizing and again
  before encoding. A committed command buffer can't be stopped, so cancellation after that point
  takes effect when the pass ends (at most a few hundred milliseconds on a phone, for a large
  batch). Splitting a long pass into several command buffers to check in between is possible but
  costs a wait per split; not planned unless a 512-token batch proves too slow to wait out.
- **Swift 6 language mode, strict concurrency on.** The package should compile with no
  `@unchecked Sendable` outside the engine's buffer wrapper.

## Architecture

### Package layout

A separate repository (working name `LayaKit`), not a directory inside laya-go, so that Swift
users get a normal SwiftPM URL and laya-go's Go module stays free of Swift files.

```
LayaKit/
  Package.swift
  LAYA_GO_COMMIT                     the laya-go commit the shaders and fixtures came from
  scripts/sync-from-laya-go.sh       copies shaders and fixtures, regenerates ShaderSource.swift
  Sources/
    LayaCore/                        text side and engine protocol; Foundation only
      Tokenizer/                     Tokenizer.swift, PreTokenizer.swift, AddedTokens.swift
      Prompt/                        PromptBuilder.swift (build_sequence, render_options)
      PyJSON/                        Parser.swift, Dumps.swift, FloatRepr.swift
      Calibration/                   Calibrate.swift, NumpyFloat32.swift
      Checkpoint/                    Config.swift (rl_agent_config.json, encoder/config.json)
      Safetensors/                   Safetensors.swift
      Engine.swift                   LayaBatch, LayaEngineOutput, LayaEngine
    LayaMetal/                       the GPU engine
      MetalEngine.swift              device, weight upload, workspace
      Forward.swift                  one command buffer per call
      SelfTest.swift                 known-answer GPU check before use
      ShaderSource.swift             generated: shaders.metal as a string literal
    LayaHub/                         download into a Hugging Face-style cache
      HubDownloader.swift, BackgroundDownloads.swift, Storage.swift
    LayaKit/                         the public façade: LayaModel, options, answers, cache
      LayaModel.swift, Question.swift, Answer.swift, AnswerCache.swift, Wire.swift
    LayaCPU/                         later (phase 5): Accelerate engine for macOS without Apple7
    laya-swift/                      executable: predict, bench, golden verify (macOS)
  Tests/
    LayaCoreTests/                   tokenizer, prompts, JSON, calibration vs fixtures
      Resources/golden/              copied from laya-go testdata/golden
    LayaMetalTests/                  logits vs fixtures, traces, self-test, stress, leaks
    LayaKitTests/                    end to end, answer cache, lifecycle
  Examples/
    LayaDemo/                        SwiftUI app (Xcode project), iOS and macOS
```

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LayaKit", targets: ["LayaKit"]),
        .library(name: "LayaCore", targets: ["LayaCore"]),   // for tools that only tokenize
        .executable(name: "laya-swift", targets: ["laya-swift"]),
    ],
    targets: [
        .target(name: "LayaCore"),
        .target(name: "LayaMetal", dependencies: ["LayaCore"]),
        .target(name: "LayaHub", dependencies: ["LayaCore"]),
        .target(name: "LayaKit", dependencies: ["LayaCore", "LayaMetal", "LayaHub"]),
        .executableTarget(name: "laya-swift", dependencies: ["LayaKit"]),
        .testTarget(name: "LayaCoreTests", dependencies: ["LayaCore"],
                    resources: [.copy("Resources/golden")]),
        .testTarget(name: "LayaMetalTests", dependencies: ["LayaMetal"],
                    resources: [.copy("Resources/golden")]),
        .testTarget(name: "LayaKitTests", dependencies: ["LayaKit"],
                    resources: [.copy("Resources/golden")]),
    ],
    swiftLanguageModes: [.v6]
)
```

No third-party dependencies. NFC normalization comes from Foundation, hashing from CryptoKit,
downloads from URLSession.

Platforms: iOS 17 and macOS 14 are a choice, not a Metal requirement. What the code needs:

| Feature | Needs | Notes |
|---|---|---|
| `simdgroup_matrix` loads, stores and multiply-accumulate (GEMM, attention) | Apple7 GPU family; MSL 2.3, which is iOS 14 / macOS 11 | The reason for the A14 / M1 floor. Checked at run time with `device.supportsFamily(.apple7)`, as laya-go does |
| `MTLGPUFamily.apple7`, `supportsFamily(_:)` | iOS 14 / macOS 11 (the enum case); iOS 13 / macOS 10.15 (the method) | |
| `MTLCompileOptions.mathMode = .safe` | iOS 18 / macOS 15 | On iOS 17 / macOS 14, `fastMathEnabled = false`, which is what laya-go falls back to |
| `makeLibrary(source:options:)` | every supported OS | Runtime compile, as laya-go |
| `DispatchSerialQueue` as an actor's executor | iOS 17 / macOS 14 | The engine's executor |
| Observation (`@Observable`) | iOS 17 / macOS 14 | Sample app and an optional `LayaLoader` view model |
| `com.apple.developer.kernel.increased-memory-limit` | iOS 15 | See [iOS](#ios-and-ipados) |

Supporting iOS 16 or macOS 13 would mean a custom executor instead of `DispatchSerialQueue` and
`ObservableObject` instead of Observation; possible, but not worth it unless a user asks
(**Estimate** of demand, not measured). Every device that runs iOS 17 but has no Apple7 GPU
(iPhone XS, XR, 11, SE 2nd generation; A12/A13 iPads) gets `unsupportedGPU` from the Metal engine.

### Shaders: compile at run time, as laya-go does

laya-go embeds `shaders.metal` with `//go:embed` and compiles it with `newLibraryWithSource` at
load, with fast math off. The options in Swift:

| Option | How | For | Against |
|---|---|---|---|
| **Source string, compiled at load (proposed)** | `sync-from-laya-go.sh` writes `ShaderSource.swift` with the file as a raw string literal (`#"""…"""#`); `device.makeLibrary(source:options:)` | Identical to laya-go: same source, same compile options (`mathMode = .safe` or `fastMathEnabled = false`). Works under `swift build`, `swift test` and Xcode alike. No `Bundle.module` | Compile cost at every load (in laya-go's app, loading including the compile took 96–670 ms on the M4 Pro). On an iPhone, **Unverified**: measure in phase 3 |
| `.metal` file processed by SwiftPM | put `shaders.metal` in the target; Xcode builds compile it into `default.metallib` in the resource bundle; `device.makeDefaultLibrary(bundle: .module)` | No compile at run time | SwiftPM gives no way to set Metal compiler flags ([swift-package-manager#5823](https://github.com/swiftlang/swift-package-manager/issues/5823)), and the Metal compiler's default is fast math on, which would change the GELU, exp and LayerNorm numerics laya-go measured with precise math. Command-line builds have had trouble finding or building `default.metallib` ([mlx-swift#460](https://github.com/ml-explore/mlx-swift/issues/460)). Both **Unverified** for current toolchains |
| Precompiled `.metallib` checked in | `xcrun metal -fno-fast-math` + `xcrun metallib` in the sync script, shipped with `.copy` | Fast load; explicit flags | A binary to keep in step with the source; one per platform (iOS and macOS metallibs differ); harder to review |
| Source compiled once, cached in an `MTLBinaryArchive` | runtime compile on first launch, archive saved to Caches | Later loads skip the compile | More code; archives can be invalidated by OS updates (**Unverified**) |

Start with the source string. If phase 3 shows the compile costs more than about 300 ms on an
iPhone, add the precompiled metallib (with `-fno-fast-math`, and a test that its logits equal the
runtime-compiled ones bit for bit).

Never download shaders. The source is part of the app binary; only weights are downloaded (see
[App Store](#app-store)).

### The Metal engine in Swift

What `internal/metal` does, and what changes in Swift:

| laya-go (`internal/metal`) | Swift |
|---|---|
| `device_darwin.go`: `MTLCreateSystemDefaultDevice` through purego, `supportsFamily:` Apple7, a command queue, compile the library, 10 pipeline states (`gemmd_h_4_4`, `gemmd_h_4_4_pf`, `gemmd_f_2_4_pf`, `embed_h`, `embed_f`, `layernorm`, `rowdot`, `act_prep`, `attention`, `gather`) | The same, with `MTLCreateSystemDefaultDevice()`, `makeComputePipelineState(function:)`. Template kernels are instantiated by `host_name` in the shader, so the names are unchanged |
| Buffers: `storageModeShared` for everything; weights with `hazardTrackingModeUntracked` | `device.makeBuffer(length:options: [.storageModeShared, .hazardTrackingModeUntracked])` for weights, `[.storageModeShared]` for activations. Shared is the natural mode on unified memory, iOS and macOS alike |
| Weights: fp16 Linear weights uploaded as stored; Wqkv rows permuted for the fused RoPE epilogue (`ropePerm`), Wi rows interleaved for the fused GeGLU epilogue (`gegluPerm`); `act_head.0` padded to a row stride of 1056; every vector (norms, biases, type embeddings) in one fp32 buffer at 16-byte-aligned offsets | Port as is, with `withUnsafeMutableBytes` on the buffer contents. The permutations are load-time row copies; getting them wrong shows up immediately in the trace test |
| Shape and dtype check of all 206 tensors, refusing unexpected ones (`load_state_dict`'s strict check) | Port as is. A checkpoint that doesn't match fails at load, never with wrong answers |
| RoPE cos/sin tables for both θ, computed in float32 the way transformers does | Port as is: `Float` math with `Foundation.cos`/`sin` on `Double`, rounded once |
| Workspace grown on demand, with 128 spare rows on every GEMM input (the kernel reads whole tiles) | Port as is; the padding is required, not an optimization |
| `Forward`: one command buffer, one compute encoder, every dispatch, one commit, one `waitUntilCompleted`; status checked | Same. `MTLCommandBuffer.status == .completed` or throw with `error` |
| Fused epilogues: bias, residual add, ReLU, GELU, GeGLU, RoPE in the GEMM | In the shader; the host only sets flags. No change |
| Flash-style attention over packed sequences with the sliding window, per (32-query block, head) | In the shader; the host writes the block table. No change |
| Last head layer only on the rows read: after its attention, only each sequence's `[CLS]` row and marker rows go through `out_proj`, the feed-forward, the scorer and the act head | Port as is (about 2.5% of the work, and fewer rows to copy back) |
| `selftest_darwin.go`: a 70×64×48 GEMM with small integers (exact in fp16 and fp32) and a 40-token windowed attention against float64 references, refusing a GPU that claims Apple7 but computes wrongly (a VM's paravirtual GPU) | Port as is. It costs a few milliseconds and is what makes `.auto` safe |
| Autorelease pools and locked OS threads around every call | Not needed in Swift: ARC, and the engine runs on its own queue. Wrap the encode in `autoreleasepool {}` on macOS to be safe with Objective-C temporaries |
| `Trace` hook: waits for the GPU after each layer and hands back named intermediates | Port as is, behind an internal test-only API. It is how a parity failure is bisected |

The `run` helper in `forward_darwin.go` (`pipeline`, `buf`, `params`, `dispatch`, `gemmOp`) maps
to a small `struct Encoder` wrapping `MTLComputeCommandEncoder` with `setBytes` for parameters.

One kernel caveat: the GEMM uses no threadgroup memory and loads 8×8 fragments straight from
device memory. Its comment says Apple9-family GPUs (A17 Pro, M3, M4) cache those loads well. It
was only measured on an M4 Pro. On Apple7 and Apple8 (A14–A16, M1, M2) the same kernel is correct
(the self-test and parity tests will say), but it may be slower than a threadgroup-tiled variant
(**Unverified**). Phase 3 measures it; a per-family kernel choice can follow.

### CPU fallback and the Simulator

Where the Metal engine doesn't start:

| Where | Why | Proposed |
|---|---|---|
| iOS Simulator | Simulator's Metal device offers capabilities like the Apple2 GPU family, so `supportsFamily(.apple7)` is false and `simdgroup_matrix` isn't available | No local engine. For development, point the app at `laya-server` on the host Mac (the Simulator reaches the Mac's `127.0.0.1`) through an `HTTPEngine`, or use a stub engine with fixture logits for previews and UI tests |
| A12/A13 iPhones and iPads on iOS 17/18 | No Apple7 | `unsupportedGPU`. A CPU engine would need fp32 weights (1.6 GB resident with Accelerate's `cblas_sgemm`), too much for 3–4 GB devices |
| Intel Macs on macOS 14 | No Apple GPU family | Optional `LayaCPU` engine (phase 5): a port of `internal/modernbert` with Accelerate. 1.7–1.8 GB, 46 ms for one state on the M4 Pro's CPU (laya-go, 2026-10-01); Intel speed **Unverified** |
| Apple silicon Mac VMs | Paravirtual GPU often fails the self-test | `LayaCPU` under `.auto`, with `engineFallback` saying why, as laya-go does |

Other fallbacks considered:

- **MPSGraph.** laya-go's unmerged `metal-graph` branch has a complete MPSGraph engine. In fp32 it
  matched Python on 99.97% (`predict_many`) and 99.95% (`predict`) of 4-decimal fields and on all
  7,081 choices, at about the custom kernels' speed on the M4 Pro, but with 1.65 GB of fp32
  weights. Its `fp16w` mode keeps weights fp16 and widens them in the graph (fp32 products); the
  branch claims fp32's answers at fp16's memory, but slower (**Unverified**: no end-to-end numbers
  recorded for `fp16w`). MPSGraph runs on pre-Apple7 GPUs, so `fp16w` is the most plausible GPU
  fallback for A12/A13 devices; whether it is fast enough there is unknown.
- **BNNS / BNNSGraph** (Accelerate) with fp16 weights: would avoid the fp32 copy on the CPU.
  Precision behavior and speed **Unverified**.
- **Swift's own loops** (a port of the portable Go kernels): correct but 5–10 times slower than
  Accelerate. Only for tests.

The CPU engine's one Go-specific hazard doesn't carry over: laya-go blocks `SIGURG`/`SIGPROF`
around `cblas_sgemm` because Go's preemption signal corrupted results mid-call. Swift has no such
signals. The other lesson does: never let `sgemm` accumulate onto the residual (`beta = 1`); add
the residual after, or ModernBERT's outlier channels cost about 4× in logit error.

## Porting the text side

Every piece below has a Go file to port line by line and a fixture that proves it.

### Tokenizer

Laya needs one operation: `encode(text) → ids` with no special tokens. laya-go's
`internal/tokenizer` (444 + 121 lines) implements exactly the configuration in
`tokenizer/tokenizer.json`, and refuses others at load:

1. **Added tokens first**, leftmost-longest: tokens with `normalized: false` (the specials) on the
   raw text, then the others on the NFC-normalized remainder. These include 23 runs of 2–24
   spaces (`"a  b"` → `a`, 50276, `b`, not `Ġ` merges), `[SEP]`/`[CLS]` (a `[SEP]` typed in a state
   becomes the real separator, 50282), `|||EMAIL_ADDRESS|||` and the like, with `lstrip`/`rstrip`
   absorbing neighbouring whitespace.
2. **NFC normalization.**
3. **ByteLevel pre-tokenizer** with the GPT-2 pattern
   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, as a hand-written
   scanner (`pretokenize.go`), then GPT-2's bytes-to-unicode map (`Ġ` for space, `Ċ` for newline).
4. **Byte-level BPE** with 50,009 merge ranks, lowest rank first, leftmost on ties, with a per-word
   cache.

**Port it; don't adopt a general library.** The options:

| Option | Parity on this tokenizer | Notes |
|---|---|---|
| **Port `internal/tokenizer` (proposed)** | laya-go's is identical on all 6,511 `tokenizer.jsonl` strings and 390 requests | About 600 lines of Swift; no dependency; the scanner needs no regex engine |
| [swift-transformers](https://github.com/huggingface/swift-transformers)' `Tokenizers` | **Unverified** | Reads `tokenizer.json` with BPE, ByteLevel and added tokens. Its pre-tokenizer uses `NSRegularExpression` (ICU), whose `\s`, `\p{L}` and `\p{N}` tables come from the OS's ICU, not Oniguruma's. Adds a dependency (and, in recent versions, Jinja and Hub code). Worth running against `tokenizer.jsonl` once as a cross-check |
| system-one-foundation-models' `ModernBERTTokenizer` | No (see [Related work](#related-work-system-one-foundation-models)) | Whitespace split plus greedy longest match over the vocabulary; no byte-level mapping, merges or added-token pass |

Swift-specific traps:

- **Work on Unicode scalars and UTF-8 bytes, never on `Character`.** A Swift `Character` is a
  grapheme cluster: `"\r\n"` is one `Character`, and so is `"]"` followed by a combining accent.
  The pre-tokenizer pattern, the added-token matcher and BPE are all defined on code points and
  bytes. Iterate `text.unicodeScalars` or `text.utf8`, and index with UTF-8 offsets.
- **Swift `String ==` uses canonical equivalence.** `"é" == "e\u{301}"` is `true` in Swift and
  `False` in Python. Never use `String` equality or `Dictionary<String, …>` lookups on text that
  may not be normalized. The added-token matcher should compare UTF-8 bytes; the BPE vocabulary is
  safe to key by `String` only because its keys are byte-level strings made of the bytes-to-unicode
  alphabet (no combining marks in the mapped text: **Unverified**; key by `[UInt8]` to be sure).
- **NFC.** Foundation's `precomposedStringWithCanonicalMapping` uses the OS's Unicode tables.
  tokenizers 0.23.2 normalizes with older tables than Python's and Go's `x/text`; the fixtures
  leave out the strings where those disagree (none in the current set, and 1 in a 20,000-string
  random run, involving combining marks from recent Unicode versions). Swift on iOS 17 and on iOS
  27 may give different results for such strings. Expect identical ids on the fixtures, and treat
  a difference on a brand-new combining mark as a known limitation.
- **`\s` is the Unicode White_Space property** (what Go's `unicode.IsSpace` reports), and `\p{L}`,
  `\p{N}` are general categories. Swift's `Unicode.Scalar.Properties.isWhitespace` and
  `generalCategory` give these; `Character.isWhitespace` and `Character.isLetter` don't (they are
  grapheme-level and use different definitions).
- **The lookahead `\s+(?!\S)`** means: a run of whitespace, minus its last character when a
  non-space follows. Port `matchAt` from `pretokenize.go` exactly rather than re-deriving it.

The gate: identical ids on `tokenizer.jsonl` (6,511 strings) and identical ids and markers on every
request in `games.jsonl` and `edge.jsonl`.

### Prompt building

`internal/prompt/prompt.go` is 157 lines: `ChoiceOption`, `ScoreOption`, `NoulOptions`,
`Builder.Build`. Ports directly. Two traps:

- **`[MASK]` replacement must be literal, on scalars or bytes.** Python's `str.replace("[MASK]",
  " ")` matches the code points. Swift's `String.replacing(_:with:)` matches `Character`s, so it
  misses `[MASK]` when a combining mark follows `]`; Foundation's `replacingOccurrences(of:with:)`
  without `.literal` uses non-literal comparison. Do it on `utf8` bytes, as Go's
  `strings.ReplaceAll` does.
- **Truncation is on token ids**, then markers past `max_len` are dropped, and a question whose
  markers didn't all fit is an error (`optionsTooLong`), not a silent drop.

### Python-compatible JSON

Structured states and non-string criteria are serialized with `json.dumps` before tokenizing, so
the bytes must match Python's. `internal/pyjson` (361 + 173 lines) parses JSON like `json.loads`
and writes it like `json.dumps`. Foundation can't do either:

- `JSONSerialization` and `Codable` don't keep object key order. Python dicts do, and the order is
  in the prompt.
- Python keeps a repeated key at its first position with its last value; integers are arbitrary
  precision (`1e22` is a float but `10000000000000000000000` is an int, written back digit for
  digit); `NaN`, `Infinity` and `-Infinity` are accepted.
- `json.dumps` writes `", "` and `": "`, keeps non-ASCII (`ensure_ascii=False`) for states, and
  escapes only `"`, `\` and control characters, with `\uXXXX` for the rest of C0. It doesn't
  escape `/`, `<`, `>` or `&`, and it writes U+2028 literally.
- **Floats use Python's `repr`**: the shortest digits that round-trip, laid out with an exponent
  when the exponent is below −4 or at least 16 (`1e-05`, `1e+16`, `0.0001`, `123.0`, `-0.0`).
  Swift's `Double.description` also gives shortest round-trip digits, but its layout rules
  differ (**Unverified** for every case; e.g. where it switches to exponent form). Take the digits
  from Swift and lay them out with laya-go's rules in `dumps.go`. `pyjson/testdata/floats.json`
  and `dumps.jsonl` are the fixtures.
- **Duplicate keys and dictionary semantics need byte equality**, for the canonical-equivalence
  reason above: `{"é": 1, "é": 2}` has two keys in Python and must keep both.

### Calibration

`internal/calibrate` (182 + 104 lines) follows numpy's float32 arithmetic step by step: float32
logits divided by a temperature cast to float32, float32 `exp` and `log`, numpy's pairwise
summation order (blocks of 8 accumulators up to 128 elements, then halves), and Python's
`round(x, 4)` (round the exact binary value half to even at 4 decimals, keeping `-0.0`).

- **Port it with `Float`, not `Double`.** The answers are rounded to 4 decimals, so matching the
  float32 steps is what makes rounded answers agree.
- **`exp` may match better in Swift than in Go.** numpy's float32 `exp` calls the platform's
  `expf`. The fixtures were generated on macOS arm64. Go can't call `expf` without cgo, so laya-go
  uses `float32(math.Exp(float64(x)))`, which differs from macOS's `expf` by one ulp for about 0.6%
  of inputs and changed a 4-decimal answer in 2 of 200,000 random questions (no fixture hits it).
  Swift's `Foundation.exp(_: Float)` calls Darwin's `expf` directly, so on macOS it should match
  numpy exactly (**Unverified**: repeat laya-go's 200,000-question comparison against numpy). Whether
  iOS's libm `expf` is bit-identical to macOS's is also **Unverified**; the on-device fixture run
  will show it.
- **Rounding.** Python's `round(x, 4)` on a float is the correctly rounded decimal. laya-go formats
  with `strconv.FormatFloat(x, 'f', 4, 64)` and parses back. In Swift, `String(format: "%.4f", x)`
  on Darwin is correctly rounded from the exact binary value (**Unverified**; `calibration.jsonl`
  checks it), then `Double(_:)`. `(x * 10000).rounded() / 10000` is wrong for some inputs, as
  laya-go's design notes.
- **Temperatures come from the checkpoint**: the bucket `"<type>:<size>"` (size `2`, `3-5`,
  `6-10`, `11+`), else the per-type value, clamped to [0.5, 5.0]. `choice:11+` (0.1006) is clamped
  to 0.5, and `info.clamped` says so.

The gate: answers computed from Python's logits are identical JSON text on `calibration.jsonl`
(264 questions), `games.jsonl` and `edge.jsonl`, both answer shapes.

### Checkpoint and safetensors

- `internal/checkpoint` reads `rl_agent_config.json` (`max_len`, `head_max_len`, temperatures) and
  lists the files a checkpoint directory must have. Port with `Codable`; these files have no
  order-sensitive content.
- `internal/safetensors` reads an 8-byte length, a JSON header (21,536 bytes in this file) and raw
  little-endian tensors. laya-go streams tensors with `ReadAt` instead of mapping the file, because
  page-cache pages don't count as the process's memory.
- **In Swift, memory-map the file** with `Data(contentsOf:options: .alwaysMapped)` (or `mmap`) and
  copy each tensor into its `MTLBuffer`. On iOS, clean file-backed pages don't count toward the
  app's memory footprint, so the mapping costs nothing toward the jetsam limit once the copy is
  done (**Unverified**: measure `phys_footprint` during and after load). Copy in parallel, as
  laya-go does with 8 workers.
- **Zero-copy is possible later, not now.** `makeBuffer(bytesNoCopy:length:)` over the whole
  mapped file (page-aligned base, tensors addressed by offset) would skip the 0.84 GB copy. It
  conflicts with the load-time row permutations for the fused RoPE and GeGLU epilogues and the
  padded `act_head.0`. Either the shaders index through the permutation, or the package writes a
  permuted copy of the file once. Phase 5, if load time or peak memory at load needs it.

## Weights and download

The checkpoint is five files at the pinned revision. `model.safetensors` is 842,609,210 bytes
(804 MiB): 205 fp16 tensors and one unused fp32 `temperature`, 421,293,830 parameters.

**Not in the app bundle.** An 843 MB bundle would put the app over the cellular download limit,
make every app update re-download the weights (**Unverified** whether App Store delta updates
would avoid that), and count against the user's storage twice during updates. Download at first
use instead.

### Download

`LayaHub` ports `hub/hub.go`:

- **Pinned revision.** `https://huggingface.co/<repo>/resolve/<revision>/<file>`, with the revision
  laya-go and LayaKit were tested against. `main` only by explicit option.
- **Integrity.** A `HEAD` request without following the redirect gives `X-Linked-Etag` (the sha256
  of an LFS file) and `X-Linked-Size`; for the small non-LFS files, the `ETag` is the git blob
  sha1. Verify after download with CryptoKit's `SHA256` / `Insecure.SHA1`, streaming the file in
  1 MB chunks. Hashing 843 MB takes a couple of seconds on a phone (**Estimate**).
- **Tokens.** An optional token goes only to `huggingface.co`, never to the CDN host the resolve
  URL redirects to. The public repo needs no token. Never ship a token in an app.
- **Foreground downloads** use `URLSession.download` with a `Range` resume from a `.incomplete`
  file, as laya-go does, and report progress per byte.
- **Background downloads** use a background `URLSessionConfiguration` so the download continues
  when the user leaves the app. Differences from the foreground path:
  - The system owns the transfer. Resuming after a failure uses the `resumeData` URLSession hands
    back, which works only when the server supports byte ranges with a stable validator. The
    resolve URL redirects to a signed CDN URL that expires, so a resume after a long gap may fail;
    then restart from the resolve URL. Whether HF's current CDN (Xet-backed storage) gives usable
    `resumeData` is **Unverified**.
  - The delegate's `didFinishDownloadingTo` must move the file synchronously before returning.
  - Relaunch handling: `application(_:handleEventsForBackgroundURLSession:completionHandler:)` on
    iOS, or SwiftUI's `.backgroundTask(.urlSession(…))`.
- **Cellular.** `allowsCellularAccess` and `allowsExpensiveNetworkAccess` off unless the app opts
  in; show the size before starting.
- **Free space.** Check `volumeAvailableCapacityForImportantUsage` for at least 1.7 GB (the file
  plus a partial copy) before starting.

### Where the files live

```
Application Support/LayaKit/hub/models--convaiinnovations--laya/
    blobs/<sha256>                    file contents, named by etag
    snapshots/<revision>/<path>       symlinks into blobs/ (or copies)
    refs/main
```

- **Application Support, excluded from backup** (`URLResourceValues.isExcludedFromBackup = true`
  on the `hub` directory). Re-downloadable data must not go to iCloud backup; Caches would be
  purged by the system under storage pressure and force another 843 MB download at an
  unpredictable time. Offer `cacheDirectory` for apps that prefer Caches.
- **The Hugging Face cache layout** keeps the directory usable by laya-go and Python tools
  pointed at it (`HF_HUB_CACHE`), and lets a Mac app reuse `~/.cache/huggingface/hub` when it is
  not sandboxed.
- **macOS sandbox:** the app's container is the default. A sandboxed app can't read
  `~/.cache/huggingface` without the user granting the folder (an open panel and a
  security-scoped bookmark). Downloads need `com.apple.security.network.client`.

### Background Assets

Apple's Background Assets framework can download large assets around install time, before the
first launch, through an app extension (iOS 16.1 / macOS 13 and later). Newer OS releases add
Apple-hosted asset packs served by the App Store (details and availability **Unverified** here).
Either would make the model ready when the user first opens the app. Costs: an extension target,
the framework's size and scheduling rules, and, for Apple hosting, redistributing the weights
(allowed under Apache-2.0 with the license and notices). Plan: URLSession first (phase 2); add a
`LayaHub` entry point a Background Assets extension can call, so apps can choose.

### App Store

- **Weights are data, not code.** App Review Guideline 2.5.2 forbids downloading code that changes
  the app's features. The model weights are parameters fed to kernels that ship in the binary; the
  Metal source is compiled from the binary, never from the network. Many shipping apps download
  ML models. This is a reading of the guidelines, not a ruling.
- **Privacy.** No data leaves the device for inference. The only network traffic is the weights
  download from huggingface.co, which should be disclosed in the app's privacy details as needed.
- **Third-party notices** for LayaKit, laya-go, Laya and ModernBERT go in the app's
  acknowledgements (see [Licensing](#licensing)).

### Smaller weights, later

| Form | Download | Resident | Status |
|---|---|---|---|
| fp16 as shipped | 842.6 MB | ~0.95 GB working set on Metal | v1 |
| int8 weight-only, per channel or per block of 64, dequantized in the GEMM load | ~0.42–0.61 GB if hosted | ~0.55 GB (**Estimate**) | Later, opt-in |
| 4-bit | ~0.25 GB | ~0.35 GB (**Estimate**) | Not planned |

Parity caveats, from laya-go's design: dynamic int8 (quantizing activations too) "destroys this
model" because ModernBERT has outlier activation channels (69–77% argmax agreement reported);
weight-only int8 was reported at 100% argmax agreement; laya-mlx found 8-bit changed predictions
and calibrated probabilities. An int8 path must pass the looser fp16/int8 tolerances (logits
≤ 5e-2, probabilities ≤ 2e-2, ≥ 99.5% choice agreement on game prompts), say so in `info.engine`,
and be a separate download we host (a modified redistribution: Apache-2.0 requires stating the
change).

## iOS and iPadOS

### Device support

Apple7 or newer GPU (**Unverified** list; check against Apple's Metal feature set tables):

| Devices | Chip | GPU family | RAM |
|---|---|---|---|
| iPhone 12, 12 mini, 12 Pro, 12 Pro Max | A14 | Apple7 | 4 GB (6 GB Pro) |
| iPhone 13 family, SE (3rd gen), 14, 14 Plus | A15 | Apple8 | 4–6 GB |
| iPhone 14 Pro, 15, 15 Plus | A16 | Apple8 | 6 GB |
| iPhone 15 Pro, 15 Pro Max | A17 Pro | Apple9 | 8 GB |
| iPhone 16 family and later | A18 and later | Apple9 or later | 8 GB+ |
| iPad (10th gen), iPad Air (4th gen) | A14 | Apple7 | 4 GB |
| iPad mini (6th gen), iPad mini (A17 Pro) | A15, A17 Pro | Apple8, Apple9 | 4–8 GB |
| iPad Air and iPad Pro with M1 or later | M1–M4 and later | Apple7–Apple9+ | 8–16 GB |

There is no `UIRequiredDeviceCapabilities` key for a GPU family (**Unverified**), and no iOS
version requires A14 (iOS 26 still supports A13), so the App Store can't exclude older devices for
us. The app must check at run time (`LayaModel.isSupported` → `supportsFamily(.apple7)` plus memory)
and explain or hide the feature.

### Memory

- **Working set about 0.95 GB** on Metal (laya-go: 0.90–0.98 GB peak resident on the M4 Pro; the
  System 1 Arcade app peaked at 1.01–1.04 GB). That is 0.84 GB of fp16 weights and embeddings plus
  activations that grow with the batch (about 30 MB for a 784-token batch, **Estimate**).
- **Jetsam.** iOS kills a foreground app that exceeds its memory limit. The limit depends on the
  device and isn't documented; commonly reported values are around half of RAM on 4 GB devices
  (**Unverified**). On a 4 GB iPhone 12, 0.95 GB for the model plus the app's own UI may be close.
- **`com.apple.developer.kernel.increased-memory-limit`** (iOS 15+) raises the limit on supported
  devices. Recommend it in the sample app and the docs; it needs the capability enabled on the App
  ID.
- **Check before loading.** `os_proc_available_memory()` gives the headroom; refuse with
  `insufficientMemory` below about 1.2 GB rather than be killed during weight upload.
- **Respond to memory warnings** (`UIApplication.didReceiveMemoryWarningNotification`): let the app
  choose between closing the model (it reloads in about a second, **Estimate**) and keeping it.

### Backgrounding

- **iOS doesn't allow GPU work in the background.** A command buffer committed after the app moves
  to the background fails (with a "background execution not permitted" error status). The engine
  must not commit then: `LayaModel` tracks the scene phase (the app tells it, or it observes
  `UIApplication` notifications) and throws `notInForeground` instead of failing inside Metal.
- **A pass in flight at the moment of backgrounding** may fail; `forward` turns that into
  `notInForeground`, and the caller can retry when active.
- **Free or keep?** A backgrounded app holding 0.95 GB is an early jetsam candidate. Default:
  keep the model while the app is merely inactive, close it after entering the background (the
  weights file stays; reload is local). Configurable.

### Thermal state and battery

- Observe `ProcessInfo.thermalStateDidChangeNotification`. At `.serious`, apps should lower their
  call rate (bigger batches, rely on the answer cache); at `.critical`, pause. LayaKit reports the
  state in its metrics; the policy is the app's.
- A phone GPU running back-to-back passes will throttle within minutes (**Estimate**); phase 3
  measures a 5-minute sustained run.
- Energy: one 6-state pass is a few tens of millijoules (**Estimate**, not measured). The answer
  cache is the main energy saver for repetitive states, as in System 1 Arcade where over 99% of
  decisions came from the cache.

### macOS notes

- Every Apple silicon Mac is Apple7 or newer. Intel Macs run macOS 14 but have no Apple GPU
  family; they need `LayaCPU` or get `unsupportedGPU`.
- App Nap throttled System 1 Arcade's built-in agent until it was disabled for the agent's
  lifetime. A Mac app that runs Laya in the background should hold a
  `ProcessInfo.beginActivity(options: .userInitiated, …)` token while it works.
- macOS allows GPU work in the background, so the iOS restriction above is iOS-only.

## Parity testing

### Fixtures

laya-go's `testdata/golden` (20 MB) is the source of truth; its
[README](../testdata/golden/README.md) documents every field. The Swift package copies it rather
than regenerating it:

- **Copy, with the commit recorded** (proposed). `scripts/sync-from-laya-go.sh <laya-go checkout>`
  copies `testdata/golden`, `internal/metal/shaders.metal` and `internal/pyjson/testdata` into the
  package, regenerates `ShaderSource.swift`, and writes the laya-go commit to `LAYA_GO_COMMIT`. A
  test checks the shader source's sha256 against the value recorded at sync time.
- **A git submodule** would avoid copies, but SwiftPM resources must live inside the target
  directory, and a submodule makes every package user clone laya-go too (SwiftPM fetches
  submodules of dependencies). Not worth it for 20 MB.
- **Regenerating** happens in laya-go only, when the Python reference or the checkpoint changes:

  ```sh
  PY=../System1Arcade/.venv/bin/python    # laya 0.3.6, torch 2.14.0, transformers 5.17.0, tokenizers 0.23.2, numpy 2.5.3
  $PY tools/make_golden.py capture        # replays System 1 Arcade's games, writes corpus/games.jsonl.gz (~1 min)
  $PY tools/make_golden.py generate       # runs the reference on CPU fp32, writes the rest (~30 min)
  ```

  Then sync both ports. Never regenerate fixtures to make a Swift test pass.

What each fixture checks in Swift:

| Fixture | Records | Swift test |
|---|---|---|
| `tokenizer.jsonl` | 6,511 strings | ids identical |
| `games.jsonl`, `edge.jsonl` | 343 + 47 requests, 3,670 sequences | `ids`, `markers`, `n_options`, errors identical; answers from Python's `logits` identical; engine logits vs `logits` (batched) and `logits_single` (per state) |
| `calibration.jsonl` | 264 questions | answers from synthetic logits identical JSON, both shapes |
| `trace/*.safetensors` | 5 sequences, incl. a 195-token one where the sliding window matters | every intermediate (`emb`, `layer.0`–`3`, `layer.27`, `final_norm`, `head_in`, `head.0`, `head.1`, `logits`, `act_logits`), to bisect a mismatch |
| `pyjson/testdata` | floats, dumps | float repr and `json.dumps` bytes identical |

Logits in the fixtures are float32 written as shortest round-trip decimals: parse with
`Float(_:)`, not `Double`.

### Tolerances

| Check | Required |
|---|---|
| Token ids, markers, option counts, `input_tokens`, error cases | identical |
| Answers from Python's logits, after 4-decimal rounding | identical JSON text |
| Raw logits vs `logits` and `logits_single`, every sequence | max abs difference ≤ 1e-3 (laya-go's Metal engine: 1.4e-4) |
| Act logits | relative difference ≤ 1e-4 |
| Every argmax (choice, top score level, noul > 0.5) | the same as Python's on every fixture (laya-go's Metal engine: all agree). A near tie (top-two gap ≤ 1e-3) that flips must be reported, not ignored |
| 4-decimal answers end to end, every field of `predict_many` and `predict` | ≥ 99.5% identical, the rest within 1e-4 (laya-go's Metal engine: 99.94% and 99.95%) |
| Trace intermediates | within laya-go's trace-test tolerances |
| **Swift Metal vs laya-go Metal, same Mac** | logits bit-identical (**Unverified** expectation; any difference means a dispatch or upload differs) |

### Running the tests

- **Swift Testing** (`import Testing`) for new tests, with parameterized tests over fixture
  records; XCTest only where Swift Testing lacks something (performance metrics with
  `XCTMetric`, `measure {}` for benchmarks).
- **macOS:** `swift test`. Model tests find the checkpoint through `LAYA_CHECKPOINT`, else the
  Hugging Face cache, and skip when it's missing; they never download, as in laya-go.
  `LAYA_GOLDEN_ALL=1` runs every record instead of a sample.
- **iOS device:** a test host app in the Examples project. The checkpoint isn't in the test
  bundle (843 MB). The host app downloads it once into its container on first run, and later
  test runs reuse it as long as the app isn't deleted; or copy it in with
  `xcrun devicectl device copy to …` (**Unverified** exact invocation). Pass environment variables
  with the scheme or `TEST_RUNNER_`-prefixed variables to `xcodebuild test`.
- **iOS Simulator:** text-side tests only (tokenizer, prompts, JSON, calibration, cache). The Metal
  tests skip with "no Apple7 GPU".
- **CI:** macOS runners for the text side and, where the runner has an Apple silicon GPU, the Metal
  tests (GitHub's hosted macOS runners are VMs; whether their GPU passes the self-test is
  **Unverified**). Device runs are manual before each release.

### Benchmarks to report

The same cases as laya-go, so the numbers line up:

| Case | What |
|---|---|
| `forward/1x56` | 1 state, 56 tokens |
| `forward/6x50` | 6 states, 300 tokens (Frogger, Space Invaders) |
| `forward/14x56` | 14 states, 784 tokens (Tetris) |
| `forward/1x512` | 1 state, 512 tokens |
| `load/cold`, `load/warm` | weights upload, shader compile, self-test; first launch vs later |
| `tokenize` | a game prompt, cold and warm cache |
| `sustained/6x50` | 5 minutes back to back: median per minute, and the thermal states reached |

For each: median and p99 over at least 40 runs, wall time and GPU time (`gpuStartTime`/`gpuEndTime`
on the command buffer), peak `phys_footprint` (from `task_info(TASK_VM_INFO)`), device model, chip,
GPU family, OS version, thermal state at start, and whether the device was charging.

## Performance estimate

**Estimate**: nothing has run on an iPhone or iPad. Basis:

- **Measured** on the M4 Pro (16-core GPU): 16 ms for 1×56 tokens (41 GFLOP), 54 ms for 6 states
  (221 GFLOP), 130 ms for 14 states (575 GFLOP), 90 ms for 1×512. The large batch runs at about
  4.4 TFLOP/s, so the engine is compute-bound there. The single state runs at about 2.6 TFLOP/s:
  partly fixed costs, partly reading 0.74 GB of linear weights per pass.
- **GPU throughput** scales with GPU cores × clock. An A17 Pro or A18 Pro has a 6-core GPU against
  the M4 Pro's 16, at a similar or slightly lower clock: roughly a third of the throughput. An
  A14, A15 or A16 has 4–5 cores at lower clocks: roughly a fifth to a quarter. An M4 iPad Pro has
  10 cores: roughly 60%. (Core counts are Apple's published figures; per-core throughput parity
  across generations is an assumption.)
- **Memory bandwidth** sets a floor for small batches: every pass reads about 0.74 GB of fp16
  linear weights. At roughly 50–60 GB/s on recent iPhones (**Unverified**) that is 12–15 ms per pass
  before any compute, against under 3 ms on the M4 Pro's 273 GB/s.
- **Not included:** thermal throttling (sustained runs may be 1.5–2× slower), and the possibility
  that the direct-load GEMM is less efficient on Apple7/Apple8 than on the Apple9 it was tuned on.

| Batch | M4 Pro (measured) | iPhone 15 Pro / 16 Pro (A17 Pro, A18 Pro) | iPhone 12–15 (A14–A16) | iPad Pro M4 |
|---|---|---|---|---|
| 1 state, 56 tokens | 16 ms | 45–70 ms | 70–120 ms | 25–40 ms |
| 6 states, 300 tokens | 54 ms | 150–250 ms | 250–400 ms | 90–130 ms |
| 14 states, 784 tokens | 130 ms | 0.4–0.6 s | 0.6–0.9 s | 0.2–0.3 s |
| 1 state, 512 tokens | 90 ms | 250–400 ms | 400–600 ms | 150–200 ms |
| Load (weights from flash, compile, self-test) | 0.1–0.7 s | 1–2 s | 1.5–3 s | 0.5–1.5 s |

What that means:

- **Interactive use is fine everywhere:** one question about a state in well under a second.
- **Realtime games at System 1 Arcade's pace** (6 inputs a second, 6-state decisions) fit on an
  A17 Pro or newer with the answer cache absorbing repeats, and are marginal on A14–A16.
- **Newer chips** (A19 and later) aren't in the table. Some add matrix hardware to the GPU that
  `simdgroup_matrix` code may not use; a Metal 4 tensor-ops kernel could, as a later per-family
  path (**Unverified**).

## Alternatives considered

| Option | For | Against | Verdict |
|---|---|---|---|
| **Swift port of laya-go with its Metal kernels (proposed)** | Proven kernels and fixtures; no dependencies; full control of precision; batching; iOS 17+ | A port to write and keep in step with laya-go | Recommended |
| **Core ML** (coremltools conversion, ANE or GPU) | The ANE is fast and power-efficient; Apple's tooling; no kernels to maintain | The ANE computes in fp16. laya-go's MPSGraph fp16 experiment (`metal-graph` branch: fp16 matrix-multiply inputs, everything else fp32) moved logits by up to 0.1, left only 59% (`predict_many`) and 71% (`predict`) of 4-decimal fields equal, and flipped 23 of 7,081 choices, because Laya's residual stream has outlier channels near 3e4 and rounding normalized activations to 10 bits loses too much. Running Core ML in fp32 on the GPU loses the ANE's advantage. Packed variable-length batches and per-sequence sliding-window masks fit Core ML's fixed or enumerated shapes poorly; a fixed 512-token shape makes a 56-token prompt pay for 512. A converted model is a modified redistribution of the weights | Not for parity. A good fit for apps that accept fp16 drift; that is what system-one-foundation-models does |
| **[MLX Swift](https://github.com/ml-explore/mlx-swift)** | laya-mlx (Python) reports 13.4 ms for one short question on an M3 Max in fp16, 944 MiB peak; MLX runs on iOS and macOS | A large native dependency and its own metallib; MLX needs a real GPU (no Simulator); the model would be rewritten in MLX ops, and its parity with laya-go's fixtures is unknown; laya-mlx found naive 8-bit and 4-bit changed predictions | Worth a benchmark against the port, not a replacement for proven kernels |
| **gomobile, wrapping laya-go** | Reuses all of laya-go, including its parity record; fastest route to a demo | gomobile builds with cgo, while laya-go's Metal engine reaches Metal through purego's `Dlopen` of system frameworks and the Objective-C runtime by selector. Whether purego works on iOS, and whether that passes App Review, is **Unverified**. A second runtime and garbage collector in the app (~10 MB, **Estimate**), a callback-style API to wrap in `async`, and Go toolchain steps in every app build | A possible phase-0 spike on macOS, not the product |
| **macOS-only: `laya-server` as a helper** | Works today; the same HTTP contract as System 1 Arcade | macOS only (iOS can't spawn processes); a sandboxed app needs `com.apple.security.network.server` for a listening socket and must embed and sign the helper; a second process to supervise | The right answer for a Mac app that wants Laya this week, and for Simulator development |
| **Adopt system-one-foundation-models' `LayaOnDevice`** | Exists; Foundation Models integration | Answers don't match Laya (below); OS 27 only; one Core ML call per question | Complement it instead (below) |

## Related work: system-one-foundation-models

[peterfriese/system-one-foundation-models](https://github.com/peterfriese/system-one-foundation-models)
(Apache-2.0; version 0.2.0, last commit 2026-09-30, read at `8fbd664`) bridges System One decision
models into Apple's Foundation Models framework. What it is, from reading the code (nothing was
run):

- **A Foundation Models provider.** It implements `LanguageModel` / `LanguageModelExecutor`, and
  translates `@Generable` types into System One questions: `Bool` → `noul`, enums → `choice`,
  `@Guide(.range(…))` → `score`. Apps call `LanguageModelSession.respond(to:generating:)` and get
  typed values with confidence and probabilities.
- **iOS, macOS and visionOS 27+**, Swift 6.1 package traits: `Jev` (default, the hosted TypeSafe
  API), `Laya` (on-device Core ML), `LayaServe` (HTTP to a self-hosted `laya-serve`), plus
  shorthand traits `OnDevice`, `Remote` and `All`.
- **A backend boundary** in `SystemOneCore`:

  ```swift
  public protocol SystemOneBackend: Sendable {
      func evaluate(request: SystemOneRequest) async throws -> SystemOneResponse
  }
  ```

  `SystemOneRequest` is one `state: String` with `questions: [String: SystemOneQuestion]`, and
  `SystemOneQuestion` is `.noul(instructions:)`, `.choice(instructions:criteria: [String: String])`
  or `.score(instructions:criteria: [String])`.
- **Its on-device path (`LayaOnDevice`)**:
  - `Tools/CoreMLConverter/convert_laya_to_coreml.py` traces Laya's `DecisionModel` at a fixed
    shape (1×512 tokens, 32 options) and converts it with `compute_precision=FLOAT16` (optional
    8-bit weights). `LayaCoreMLEngine` loads it with `computeUnits = .all`, so the ANE is used
    where Core ML chooses.
  - **One Core ML prediction per question**, in a loop; no batching across questions or states.
    The engine builds inputs of the sequence's own length while the model was traced at 512, so
    either Core ML pads or the shapes must match; from reading alone this looks like it may not
    run as written (**Unverified**).
  - **`ModernBERTTokenizer`** splits on whitespace and does a greedy longest match over the
    vocabulary. It has no byte-level mapping (so no `Ġ` word-initial tokens), no BPE merges, and
    no added-token pass, so its ids won't match Hugging Face's for ordinary text. Its tests use a
    small hand-made vocabulary; there are no golden or parity tests against Python.
  - **Calibration** uses fixed temperatures (1.0 for 2-option buckets, 1.2 otherwise) instead of
    the checkpoint's fitted, clamped values; confidence is `max(p)` instead of `1 − H(p)/log k`;
    rounding is `round(x * 10000) / 10000`, which differs from Python's `round(x, 4)` for some
    inputs. The act head isn't used.
  - Its mobile guide's size table gives "~210 MB" for 8-bit `laya` (421M parameters), which is
    the size of 4-bit weights; 8-bit would be about 421 MB.

So its integration layer is valuable and its on-device numerics are not Laya's. The two packages
fit together:

- **An adapter that conforms to `SystemOneBackend`**, backed by `LayaModel`, gives
  system-one-foundation-models' users laya-go-level accuracy, the real tokenizer, fitted
  temperatures and GPU batching, behind the API they already use. Mapping:
  `SystemOneQuestion.choice` → `Question.choice(_:sortedFrom:)` (it already sorts criteria keys,
  matching System 1 Arcade's order); `.score` → `Question.score`; `.noul` → `Question.noul`; one
  `predict` call per request answers every question in one pass; `SystemOneAnswer` gets Python's
  4-decimal values, with `legend` for scores.
- **Where it lives.** Options, in order of preference:
  1. **Upstream, as a trait** in system-one-foundation-models (say `LayaMetal`), with a
     trait-conditional dependency on LayaKit (SwiftPM 6.1 supports trait-conditioned target
     dependencies; **Unverified** whether that keeps LayaKit out of the dependency graph for users
     who don't enable the trait). Keeps one Foundation Models entry point for users.
  2. **A small separate package** (`LayaKitSystemOne`) depending on both, with platforms 27+.
     Needs no upstream agreement.
  3. **A target inside LayaKit** marked `@available(iOS 27, macOS 27, *)`. Drags
     system-one-foundation-models into every LayaKit user's graph. Not recommended.
- **Contributing upstream** is worth offering regardless: LayaKit's `LayaCore` tokenizer and
  calibration could replace `ModernBERTTokenizer` and `TemperatureCalibration`, fixing their Core
  ML path's text side even for users who keep Core ML; and laya-go's golden fixtures could give
  that repository parity tests. The fp16/ANE question is theirs to weigh, with the MPSGraph fp16
  evidence above.
- **Platforms.** LayaKit itself stays at iOS 17 / macOS 14: apps that don't use Foundation Models
  (or must support pre-27 devices) shouldn't need OS 27. Only the adapter needs 27.

## Licensing

- **LayaKit: Apache-2.0**, with a NOTICE file.
  - It is a translation of laya-go (Apache-2.0, Copyright 2026 Brennan Stehling), which is itself
    a translation of Laya's Python package (Apache-2.0, Convai Innovations). Swift files ported
    from a Go file say so in their header, and the NOTICE carries laya-go's NOTICE text forward.
  - `shaders.metal` is copied from laya-go verbatim: keep its header, add the Apache-2.0 notice
    and the laya-go commit, and list it in the NOTICE.
  - The NOTICE credits Laya (Convai Innovations, NandhaKishorM/laya, laya 0.3.6) for the prompt
    builder, calibration and model definition, and ModernBERT (Answer.AI and LightOn,
    Apache-2.0) for the encoder architecture.
- **The weights** (`convaiinnovations/laya`) are published under Apache-2.0 per the model card's
  front matter (the Hugging Face repo has no LICENSE file), with a `commercial-use` tag. LayaKit
  downloads them from upstream at a pinned revision and doesn't redistribute them. Hosting them
  elsewhere (Background Assets, an int8 conversion) is allowed under Apache-2.0 with the license,
  the notices, and a statement of changes.
- **system-one-foundation-models** is Apache-2.0, so an adapter or upstream contribution has no
  license friction.
- **Apps** that ship LayaKit must include the Apache-2.0 text and LayaKit's NOTICE, for example in
  an acknowledgements screen or a Settings bundle.

This is a reading of the licenses, not legal advice.

## Plan

Each phase ends with something testable. Durations are rough, for one person who knows Swift and
Metal (**Estimate**).

**Phase 0: package and text side (1–1.5 weeks)**

- Repository, `Package.swift`, `sync-from-laya-go.sh`, fixtures as test resources.
- Port the tokenizer, `pyjson`, prompt builder, calibration, checkpoint config.
- *Milestone:* identical ids on all 6,511 tokenizer strings and all 390 requests; identical
  answers from Python's logits on `games`, `edge` and `calibration`; on macOS and in the iOS
  Simulator.

**Phase 1: Metal engine on macOS (1.5–2 weeks)**

- safetensors (mapped), weight upload with the permutations, workspace, `forward`, self-test,
  trace hook. `laya-swift golden verify` and `laya-swift bench`.
- *Milestone:* every sequence within 1e-3 (expect ~1.4e-4), traces within tolerance, ≥ 99.5% of
  4-decimal fields and every argmax equal end to end; logits compared with laya-go's on the same
  Mac (bit-identical expected); speed within 10% of laya-go's benchmarks.

**Phase 2: public API, download, cache (about 1 week)**

- `LayaModel` actor, `LayaHub` (foreground and background URLSession, verification, storage),
  `LayaAnswerCache`, errors, `info`.
- *Milestone:* a fresh macOS install downloads, verifies, loads and answers; a killed download
  resumes; answers through the cache equal answers without it.

**Phase 3: iOS on device (1–1.5 weeks)**

- Memory checks and the increased-memory-limit entitlement, scene-phase handling, thermal
  reporting, memory warnings. The test host app.
- Benchmarks on at least one A14–A16 iPhone, one A17 Pro or newer iPhone, and one M-series iPad.
- *Milestone:* full fixture parity on device; the benchmark table above replaced by measurements;
  no jetsam in a 5-minute sustained run on a 6 GB iPhone; decision on a precompiled metallib.

**Phase 4: SwiftUI sample app and docs (about 1 week)**

- `Examples/LayaDemo` for iOS and macOS: a download screen with size, progress, cellular opt-in
  and cancel; a "loading model" state; a form to build `choice`/`score`/`noul` questions about a
  text state and show probabilities and confidence; the engine name and timing; lifecycle
  handling (background, memory warning, thermal state). An `@Observable` `LayaLoader` drives it.
  SwiftUI previews use a stub engine with fixture logits.
- DocC documentation; README with the parity and benchmark tables.
- *Milestone:* the app runs on an iPhone and a Mac from a clean install; first public release
  (0.1.0), recording the laya-go commit, Laya version and checkpoint revision.

**Phase 5: reach and optimizations (spikes, then pick)**

- `SystemOneBackend` adapter (upstream trait or separate package).
- `LayaCPU` for Intel Macs and VMs; a GPU fallback for A12/A13 (MPSGraph `fp16w`), if measured
  fast enough.
- Per-family GEMM for Apple7/Apple8 if phase 3 shows the direct-load kernel is slow there;
  Metal 4 tensor ops on chips with GPU matrix hardware.
- Zero-copy weights; `MTLBinaryArchive`; Background Assets.
- Opt-in int8 weight-only download, gated by the fp16/int8 tolerances.

**Phase 6: more checkpoints**

- `typed-decisions` (same architecture, `max_len` 1024, `head_max_len` 256) and `multilingual`
  (mmBERT-base, 256k-vocabulary tokenizer to verify separately), following laya-go.

Phases 0–4: about 6–7 weeks (**Estimate**).

## Risks and open questions

- **iPhone speed and thermals.** The estimate could be off by 2× either way, and sustained use
  throttles. Phase 3 measures before any promise is made.
- **Memory on 4 GB devices.** 0.95 GB plus the app may hit the jetsam limit on iPhone 12 / 13 and
  4 GB iPads, even with the entitlement. Possibly int8 weights for those devices, with their parity
  caveats, or no support.
- **Tokenizer drift in Swift.** `Character` vs scalar semantics, canonical-equivalence string
  equality and OS-dependent Unicode tables are easy to get subtly wrong; only the fixtures protect
  against it.
- **Two ports to keep in step.** A change to laya-go's kernels, prompt or calibration must reach
  the Swift port. The sync script, `LAYA_GO_COMMIT` and the shader checksum test make drift
  visible; a laya-go CI job that runs the Swift text-side tests would make it fail loudly.
- **Upstream Laya changes.** As in laya-go: pin laya 0.3.6 and the checkpoint revision per release,
  and regenerate fixtures in laya-go when either changes.
- **Shader compile cost on iPhone** is unmeasured; the precompiled-metallib path needs fast math
  off, which SwiftPM can't set.
- **Background download resumption** against Hugging Face's CDN is unverified.
- **GEMM efficiency on Apple7/Apple8** is unverified; the kernel was tuned on Apple9.
- **App Review** of downloaded weights is a reading of the guidelines, not a ruling.
- **Name and home.** LayaKit is a working name. Ask the Laya authors whether they would link it,
  and Peter Friese whether system-one-foundation-models would take a `LayaMetal` trait.
- **Should the Swift port live in laya-go instead?** One repository would keep shaders and
  fixtures in one place, but SwiftPM would then check out the Go module (and the fixtures) for
  every Swift user, and Go users would get Swift files. A separate repository with a sync script is
  proposed.
