# VAIML compile cache

NPU-compiled model artifacts (FP32 → BF16) live here — one subfolder per model,
built automatically on first run (several minutes, up to 60 min for larger
models). Binaries are git-ignored; only this
README and the portable `vitisai_config.json` are committed.

The workshop uses two models, so two subfolders are built:

- `yolo26s-pose_fp32/` — the pose model (Sections 3–7)
- `hand_landmark_fp32/` — the 21-point hand model (Section 8)

**Reusing a pre-compiled cache:** copy a model subfolder in to skip the first-run
compile, but only if OS + Ryzen AI toolchain (1.7.1) match. A mismatched cache makes
VitisAI **silently fall back to CPU**. If NPU looks as slow as CPU, wipe and recompile:

```bash
rm -rf cache/*/
```

The hand model also needs its model file at `models/hand_landmark.onnx` (bundled in
the repo; auto-downloaded on first run if missing, ~10 MB). To fully pre-provision a
machine, copy in both `cache/hand_landmark_fp32/` and that file.
