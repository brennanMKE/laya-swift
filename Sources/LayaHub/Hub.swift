import Foundation

/// The checkpoint LayaKit is tested against, matching laya-go's pinned revision.
public enum LayaCheckpoint {
    public static let repo = "convaiinnovations/laya"
    public static let revision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
    public static let files = [
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]

    /// The snapshot folder in a Hugging Face cache, if it holds every file.
    public static func cachedSnapshot(in hubCache: URL) -> URL? {
        let snapshot = hubCache
            .appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
            .appendingPathComponent(revision)
        let fm = FileManager.default
        return files.allSatisfy { fm.fileExists(atPath: snapshot.appendingPathComponent($0).path) } ? snapshot : nil
    }

    /// The default Hugging Face cache: $HF_HUB_CACHE, $HF_HOME/hub, or ~/.cache/huggingface/hub.
    public static var defaultHubCache: URL {
        let env = ProcessInfo.processInfo.environment
        if let dir = env["HF_HUB_CACHE"] { return URL(fileURLWithPath: dir) }
        if let home = env["HF_HOME"] { return URL(fileURLWithPath: home).appendingPathComponent("hub") }
        return URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".cache/huggingface/hub")
    }
}
