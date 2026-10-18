# Live classroom experiment

Measured live CUDA run; weights are excluded. Paths in manifest.json are relative to this folder.
Extract this folder under results/replays/ in the repository, review, then commit it.
This live-export format is not the historical load_replay_manifest() format.
Verify checksums with verify_export_bundle(path); comparison.json contains paired scores and bootstrap intervals.
The summary includes selected training/dev IDs and actual hardware/settings; evaluation_documents.jsonl preserves gold.
