# seisrefine

`seisrefine` is a Rust backend for conservative SEG-Y-derived volume refinement.

Current capabilities:

- SEG-Y inspection through `sgyx`
- geometry-aware ingest of regular post-stack cubes into a chunked on-disk store
- 2x inter-trace upscaling with `linear` and `cubic` methods
- validation harness with JSON reports across a fixed local corpus
- CSV section rendering for backend inspection workflows

Current ingest behavior:

- `seisrefine` asks `sgyx` for geometry analysis before dense cube assembly
- non-dense or duplicate-heavy surveys are rejected intentionally in v1
- ingest can accept explicit inline/crossline and optional third-axis header mappings

## Development

This repository expects a sibling checkout of `sgyx` in the same parent folder:

```text
dev/
  sgyx/
  seisrefine/
```

Run locally with:

```bash
cargo test
cargo run -- inspect ../sgyx/test-data/small.sgy
cargo run -- ingest ../sgyx/test-data/small.sgy ./small.zarr
cargo run -- ingest ../sgyx/test-data/small.sgy ./small.zarr --inline-byte 17 --crossline-byte 25
cargo run -- validate ./target/validation-reports
```
