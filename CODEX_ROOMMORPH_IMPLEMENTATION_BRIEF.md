# Codex Implementation Brief: RoomMorph AI Gradio MVP

## 1. Role and Objective

You are working on the existing repository:

```text
https://github.com/Ch4lalala/stable-diffusion-with-pytorch
```

The repository contains a from-first-principles PyTorch implementation of Stable Diffusion 1.5 inference, including CLIP, VAE, U-Net/diffusion, DDPM sampling, classifier-free guidance, text-to-image, and image-to-image.

Build a focused Gradio application named **RoomMorph AI** inside the same repository. The application must let a user upload a room photograph and generate an interior-design concept using the repository's existing image-to-image pipeline.

The purpose is an AI Engineer internship portfolio project. Prioritize a reliable, understandable MVP and preserve the educational value of the custom Stable Diffusion implementation.

## 2. Important Context and Constraints

- Primary local development machine: MacBook Air M3 with 8 GB unified memory.
- Local acceleration: PyTorch MPS when available.
- Planned public demo hardware: Hugging Face Spaces CPU Basic with 2 vCPU and 16 GB RAM.
- The Stable Diffusion checkpoint is approximately 4.27 GB and must never be committed to Git.
- Existing local model filename:

  ```text
  data/v1-5-pruned-emaonly.ckpt
  ```

- Existing tokenizer files:

  ```text
  data/vocab.json
  data/merges.txt
  ```

- The existing model mathematics and pretrained-weight conversion must remain unchanged unless an actual bug prevents the application from running.
- Inspect the repository, imports, current working tree, and existing code before making changes. Preserve unrelated user changes.
- Do not push, create a pull request, deploy, or make external mutations unless explicitly requested.
- Do not download or execute the 4 GB checkpoint merely to perform a code-level smoke test. Full inference verification can be left as a clearly documented manual test when the checkpoint is unavailable.

## 3. MVP Scope

Implement image-to-image only for the first release.

The user workflow is:

1. Open the Gradio application.
2. Upload a photograph of a room.
3. Select an interior style preset.
4. Optionally enter additional design instructions.
5. Adjust strength, inference steps, CFG, and seed.
6. Click **Generate Design**.
7. View the original image, generated image, and generation metadata.

Supported initial style presets:

- Japandi
- Minimalist
- Scandinavian
- Industrial
- Modern
- Cozy

Do not add text-to-image, authentication, a database, payment, an LLM, ControlNet, or inpainting in this MVP.

## 4. Expected Repository Structure

Adapt this structure to the current repository without unnecessarily moving existing files:

```text
stable-diffusion-with-pytorch/
├── app.py
├── sd/
│   ├── attention.py
│   ├── clip.py
│   ├── ddpm.py
│   ├── decoder.py
│   ├── diffusion.py
│   ├── encoder.py
│   ├── model_converter.py
│   ├── model_loader.py
│   └── pipeline.py
├── assets/
│   └── README.md
├── data/                         # ignored by Git
├── outputs/                      # ignored by Git except curated assets
├── requirements.txt
├── .gitignore
└── README.md
```

If root-level imports do not work because the existing `sd` modules use flat imports, choose the smallest maintainable fix. Avoid a broad package refactor unless necessary. The notebook or existing execution flow should not be broken.

## 5. Application Requirements

### 5.1 Device selection

Select the runtime device in this order:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Use CPU as the idle device for CUDA and MPS. Moving a model to CPU when both runtime and idle devices are CPU must remain safe.

### 5.2 Model loading

- Load the tokenizer and Stable Diffusion components only once per application process, not once per button click.
- A cached/lazy singleton loader is acceptable.
- Show a clear startup or UI status while the model is loading.
- Resolve paths relative to `app.py`, not the shell's current working directory.
- Support an optional environment variable such as `SD_MODEL_PATH` for the checkpoint location.
- When files are missing, provide a helpful error listing the expected paths. Do not crash with an unexplained traceback in the UI.
- Keep `data/` and weight files out of Git.
- Optional deployment preparation: provide a clearly isolated helper that can download the checkpoint/tokenizer from the configured Hugging Face model repository if local files are absent. Do not make this download mandatory for local code smoke tests.

### 5.3 Image preprocessing

- Accept PNG, JPG, and JPEG uploads.
- Convert input images to RGB.
- Handle transparent PNGs by compositing them onto a neutral background before RGB conversion.
- Resize/crop with `PIL.ImageOps.fit` and Lanczos resampling so the input is not stretched.
- Use dimensions divisible by 8.

### 5.4 Prompt construction

Build a concise positive prompt from:

- selected style preset;
- room/interior context;
- user's optional instruction;
- general quality descriptors.

Example intent:

```text
a realistic interior photograph of the same room, Japandi interior design,
warm neutral colors, natural wood furniture, preserve the room layout,
soft natural lighting, highly detailed
```

Use a conservative negative prompt such as:

```text
distorted room, warped walls, duplicate furniture, floating objects,
deformed architecture, blurry, low quality
```

Do not include desired positive concepts in the negative prompt.

### 5.5 Generation settings

Provide a lightweight default suitable for limited hardware:

```text
resolution: 256 x 256
inference steps: 10
strength: 0.60
CFG enabled: false by default on CPU
CFG scale: 7.5 when enabled
seed: 42
sampler: DDPM
```

Requirements:

- Let users select a supported resolution, but default to 256 x 256.
- Restrict public-demo choices to safe values; do not allow arbitrary huge resolutions or step counts.
- Suggested resolution choices: 256 and 512 only.
- Suggested inference-step range: 5 to 30, default 10.
- Suggested strength range: 0.20 to 0.90, default 0.60.
- Suggested CFG-scale range: 1.0 to 12.0, default 7.5.
- Explain in the UI that enabling CFG generally improves prompt adherence but increases computation.
- Ensure all associated pipeline latent dimensions are updated when resolution changes.
- Wrap inference in `torch.inference_mode()` or an equivalent no-gradient context.
- Seed generation reproducibly using the existing pipeline interface.

### 5.6 Gradio interface

Create a clean interface containing:

- Project title and short explanation.
- Runtime device indicator.
- Warning that free CPU inference may take several minutes.
- Input image component.
- Style dropdown.
- Additional-instruction textbox.
- Strength slider.
- Inference-steps slider.
- CFG enable checkbox.
- CFG-scale slider.
- Seed number input.
- Resolution selector.
- **Generate Design** button.
- Output image component.
- Read-only metadata/status area containing at least:
  - runtime device;
  - resolution;
  - steps;
  - effective approximate img2img steps (`steps * strength`);
  - strength;
  - CFG setting;
  - seed;
  - elapsed generation time.

Use Gradio's queue with a concurrency limit of one so multiple generations do not compete for model memory.

### 5.7 Output handling

- Return a displayable PIL image or compatible NumPy array.
- Save successful outputs under `outputs/` with timestamped, collision-resistant filenames.
- Do not track ordinary generated outputs in Git.
- Create `assets/README.md` explaining where curated portfolio screenshots should be placed later.

### 5.8 Errors and validation

- Reject generation when no input image is provided.
- Reject empty or invalid settings with a user-friendly Gradio error.
- Catch predictable file/model/runtime errors and show concise messages.
- Preserve full exception details in server logs for debugging, without exposing an overwhelming traceback in the UI.
- Restore/offload models safely after a failed generation when possible.

## 6. Lightweight and Smoke-Test Support

Add a safe way to validate imports and construct the Gradio UI without loading the checkpoint, for example:

```text
ROOMMORPH_SKIP_MODEL_LOAD=1
```

In this mode:

- The UI must launch.
- The UI must clearly say that inference is disabled.
- Pressing Generate must return a clear message rather than a fabricated AI result.
- Never present placeholder output as real model inference.

This is intended for CI and development environments without the checkpoint.

## 7. Dependencies

Create or update `requirements.txt` with only direct runtime dependencies. Inspect the working environment and existing imports before choosing compatible version constraints. Expected packages include:

```text
torch
numpy
Pillow
transformers
tqdm
gradio
huggingface_hub  # only if deployment download support is implemented
```

Do not use a blind `pip freeze` containing unrelated local packages.

## 8. Git Ignore Requirements

Ensure `.gitignore` covers at least:

```gitignore
data/
outputs/
*.ckpt
*.safetensors
__pycache__/
*.py[cod]
.venv/
venv/
.DS_Store
```

Do not remove valid existing ignore rules.

## 9. README Requirements

Update the main README without removing valuable existing documentation. Include:

- Project overview and practical problem.
- Clear statement that this is Stable Diffusion 1.5 inference implemented with PyTorch and pretrained weights, not model training from scratch.
- RoomMorph AI feature list.
- Architecture flow:

  ```text
  Room image + design instruction
               -> VAE encoder
               -> noisy latent
               -> CLIP-conditioned U-Net denoising with DDPM
               -> VAE decoder
               -> redesigned room concept
  ```

- Local installation instructions.
- Required checkpoint/tokenizer files and their expected locations.
- How to run `python app.py`.
- Device support: CUDA, MPS, CPU.
- Hardware limitations and realistic CPU warning.
- Example-gallery placeholder section.
- Known limitations: output is a visual concept, exact geometry/furniture preservation is not guaranteed, and image-to-image is not inpainting.
- License/model-weight attribution and a reminder not to commit weights.

## 10. Verification

Perform all safe checks that do not require downloading or running the full model:

1. Inspect Git status before editing.
2. Compile/import the changed Python files.
3. Launch or construct the Gradio interface using skip-model-load mode.
4. Verify required paths resolve from `__file__`, not `Path.cwd()`.
5. Confirm that `.gitignore` excludes weights, `data/`, and ordinary outputs.
6. Confirm no generated binary/model files were accidentally added.
7. If the checkpoint is already available and the user explicitly permits a long inference test, perform one lightweight 256 x 256 test. Otherwise document the exact manual command and expected behavior.

Do not claim full inference passed unless an actual generation completed.

## 11. Acceptance Criteria

The task is complete when:

- `python app.py` launches the RoomMorph AI Gradio interface.
- Skip-model-load mode launches without requiring the checkpoint.
- Normal mode loads the existing tokenizer/model once.
- A valid uploaded image is passed to the repository's real image-to-image pipeline.
- PNG/JPG preprocessing is correct and aspect ratio is not stretched.
- Device selection supports CUDA, MPS, and CPU.
- The application defaults are safe for limited hardware.
- Only one generation can run concurrently.
- Output and metadata are shown clearly.
- Ordinary generated images and model weights remain untracked.
- README and requirements are sufficient for another developer to run the app.
- Existing Stable Diffusion modules and notebook workflows are not unnecessarily broken.

## 12. Delivery Report

At the end, report:

1. Files created and modified.
2. Important implementation decisions.
3. Commands/checks executed and their results.
4. Anything not tested, especially full model inference.
5. Exact local run command.
6. Recommended next step for Hugging Face CPU Basic deployment.

