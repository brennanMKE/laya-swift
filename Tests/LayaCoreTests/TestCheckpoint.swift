import Foundation

/// The checkpoint the model tests run against: $LAYA_CHECKPOINT, else the
/// pinned revision in the local Hugging Face cache. Tests that need it skip
/// when it's missing; they never download, as in laya-go.
enum TestCheckpoint {
    static let revision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

    static let directory: URL? = {
        let env = ProcessInfo.processInfo.environment
        if let dir = env["LAYA_CHECKPOINT"] { return URL(fileURLWithPath: dir) }
        let hub: URL
        if let dir = env["HF_HUB_CACHE"] {
            hub = URL(fileURLWithPath: dir)
        } else if let home = env["HF_HOME"] {
            hub = URL(fileURLWithPath: home).appendingPathComponent("hub")
        } else {
            hub = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".cache/huggingface/hub")
        }
        let snapshot = hub.appendingPathComponent("models--convaiinnovations--laya/snapshots/\(revision)")
        let marker = snapshot.appendingPathComponent("tokenizer/tokenizer.json")
        return FileManager.default.fileExists(atPath: marker.path) ? snapshot : nil
    }()

    static var tokenizerURL: URL? { directory?.appendingPathComponent("tokenizer/tokenizer.json") }
}
