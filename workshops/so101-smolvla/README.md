# SO-101 SmolVLA digital twin

Simulated SO-101 arm. The demo policy is SmolVLA fine-tuned on this scene.

## Environment

ROCm 10.1 userspace and PyTorch live in `.venv`. The system ROCm install is left as it is. This machine's iGPU is gfx1151, so the wheels use the `device-gfx1151` extra from `https://stable.repo.amd.com/rocm/whl-next/`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"
.venv/bin/python -m rocm_sdk test
```

The pins come from the [ROCm 10.1.0 pip install](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html) and the [PyTorch for ROCm 10.1.0](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.1.0) page for gfx1151. The environment check reads the venv packages. `/opt/rocm/.info/version` is the system stack and is not the gate.
