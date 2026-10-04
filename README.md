# LayaKit

Laya, a System 1 decision model, running on the GPU in Swift apps for iPhone, iPad and Mac. It's a
port of [laya-go](https://github.com/brennanMKE/laya-go): the same Metal shaders, and the same
golden fixtures to prove its answers match the Python reference.

**Status: bootstrapped.** The package builds for macOS and iOS, compiles laya-go's shaders on the
GPU, and loads the fixtures. It can't answer questions yet. The plan, API and phases are in
[docs/swift-port.md](docs/swift-port.md). `LayaKit` is a working name.

## Requirements

- iOS 17 or macOS 14, Swift 6
- A GPU in the Apple7 family or newer for the Metal engine: iPhone 12 (A14) or newer, or any Mac
  with Apple silicon

## Layout

| Target | What |
|---|---|
| `LayaCore` | The text side (tokenizer, prompts, Python-compatible JSON, calibration) and the engine protocol |
| `LayaMetal` | The GPU engine; `ShaderSource.swift` is generated from laya-go's `shaders.metal` |
| `LayaHub` | The pinned checkpoint and, later, downloading it |
| `LayaKit` | The public API (`LayaModel`) |
| `laya-swift` | A development tool for macOS |

## Building and testing

```sh
swift build
swift test
swift run laya-swift info     # checks the GPU, compiles the shaders, looks for the checkpoint
```

On this project's M4 Pro, `info` prints:

```
GPU: Apple M4 Pro
Shaders: 10 kernels compiled in 288 ms
Checkpoint: ~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc1…
```

## Syncing from laya-go

The shaders and fixtures come from a laya-go checkout next to this one:

```sh
scripts/sync-from-laya-go.sh [path/to/laya-go]
```

It regenerates `Sources/LayaMetal/ShaderSource.swift`, copies `testdata/golden` into the tests and
records the commit in `LAYA_GO_COMMIT`.

## Related

- [laya-go](https://github.com/brennanMKE/laya-go): the Go implementation this ports
- [System 1 Arcade](https://github.com/brennanMKE/System1Arcade): the games laya-go was built for
- [system-one-foundation-models](https://github.com/peterfriese/system-one-foundation-models):
  System One models in Apple's Foundation Models framework. See
  [Related work](docs/swift-port.md#related-work-system-one-foundation-models) for how the two fit.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
