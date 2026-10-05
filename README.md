# Embedded x86 AI

Tutorials, demos, workshops, and enablement material for AI development with ROCm and Ryzen AI on AMD embedded x86 platforms.

## Labs

| Name | Description |
|------|-------------|
| [reachymini-lab](labs/reachymini-lab/) | Reachy Mini robot that sees, listens, and answers — running fully local on AMD |

## Skills

| Name | Description |
|------|-------------|
| [cuda-to-hip-porting](skills/) | Claude Code skill that ports NVIDIA CUDA codebases to AMD HIP/ROCm end-to-end |

## Workshops

| Name | Event | Description |
|------|-------|-------------|
| [vvla-pipeline](workshops/vvla-pipeline/) | Advancing AI 2026 | Voice-vision-language-action pipeline driving a physical SO-101 arm (NPU + iGPU + CPU) |
| [vision-kernels-rocm](workshops/vision-kernels-rocm/) | Advancing AI 2026 | Write custom HIP/ROCm vision kernels feeding a simulated robot arm (iGPU) |
| [ryzen-ai-always-on](workshops/ryzen-ai-always-on/) | Advancing AI 2026 | Always-on inference on the Ryzen AI NPU — YOLO26 pose and pinch-to-teleop of a simulated SO-101 arm (MuJoCo) |

For the VVLA workshop, install Ryzen AI 1.8.0 outside this repository by following the [Linux installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html). From `~/ryzen_ai-1.8.0`, run `./install_ryzen_ai.sh -a yes -p $PWD/venv`. Then bootstrap the workshop:

```bash
export RYZEN_AI_WHEELS=~/ryzen_ai-1.8.0
cd embedded-x86-ai/workshops/vvla-pipeline
./bootstrap.sh --skip-compile
```

## Issues and Support

Open a [GitHub issue](../../issues) to report bugs or ask questions.

## License

**NOTICE:** The user acknowledges and agrees that use of certain materials in this repo requires installation of Ryzen AI software which is subject to the EULA available [here](https://ryzenai.docs.amd.com/en/latest/licenses.html) (the "Ryzen AI EULA"). The user understands that once the user compiles the open source files offered in this GitHub repo, any use of such compiled product must comply with the terms of the Ryzen AI EULA.

Workshops and labs: [BSD 3-Clause](LICENSE) — Copyright (C) 2026 Advanced Micro Devices, Inc.

