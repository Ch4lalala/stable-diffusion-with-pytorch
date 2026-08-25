# RoomMorph AI

RoomMorph AI is a focused Gradio application that redesigns a room photograph or
generates a new interior concept from text. It is built on this repository's
educational Stable Diffusion 1.5 inference implementation, so the application
uses the custom CLIP, VAE, U-Net, and DDPM code under `sd/` instead of hiding
inference behind a third-party diffusion pipeline.

This project performs inference with pretrained Stable Diffusion 1.5 weights. It
does **not** train Stable Diffusion from scratch. The practical goal is to help a
user explore an interior visual direction while keeping the model internals
understandable for an AI Engineer portfolio.

## Features

- Image-to-image generation from an uploaded PNG, JPG, or JPEG room photo.
- Text-to-image generation from positive and negative room prompts.
- Japandi, Minimalist, Scandinavian, Industrial, Modern, and Cozy presets.
- Optional room-specific design instruction.
- Safe controls for strength, inference steps, CFG, seed, and 256/512 resolution.
- CUDA, Apple MPS, and CPU device selection in that order.
- Lazy singleton model loading and one-generation-at-a-time queueing.
- Aspect-ratio-preserving crop, RGB conversion, and transparent PNG handling.
- Timestamped outputs plus runtime and generation metadata.
- Checkpoint-free UI smoke-test mode for development and CI.

Phase 2 provides separate image-to-image and text-to-image tabs backed by one
cached model and tokenizer instance. It does not include inpainting, ControlNet,
authentication, a database, payments, or an LLM.

## Architecture

```text
Room image + design instruction
             -> VAE encoder
             -> noisy latent
             -> CLIP-conditioned U-Net denoising with DDPM
             -> VAE decoder
             -> redesigned room concept

Style preset + positive/negative prompts
             -> random latent
             -> CLIP-conditioned U-Net denoising with DDPM
             -> VAE decoder
             -> new room concept
```

The application resizes and center-crops image inputs with Pillow, builds concise
style prompts, and calls the real `sd/pipeline.py` for both modes. Models and the
tokenizer are loaded once onto CPU and reused by both tabs; each component is
moved to the runtime device only when needed. Existing model mathematics and
checkpoint conversion remain unchanged.

## Repository Structure

```text
.
├── app.py
├── assets/
│   └── README.md
├── data/                         # local tokenizer and weights, ignored
├── images/                       # local notebook inputs, ignored
├── outputs/                      # ordinary generations, ignored
├── requirements.txt
├── sd/
│   ├── attention.py
│   ├── clip.py
│   ├── ddpm.py
│   ├── decoder.py
│   ├── demo.ipynb
│   ├── diffusion.py
│   ├── encoder.py
│   ├── model_converter.py
│   ├── model_loader.py
│   └── pipeline.py
└── README.md
```

## Local Installation

Python 3.10 or newer is recommended. Create an isolated environment from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When using an existing Conda environment, install and run with the same Python
interpreter so Jupyter and Gradio see the same packages:

```bash
python -m pip install -r requirements.txt
```

`pytorch-lightning` is required because the legacy `.ckpt` may reference it
while `torch.load` deserializes the trusted checkpoint.

## Required Model Files

Put these local files in `data/`:

```text
data/v1-5-pruned-emaonly.ckpt
data/vocab.json
data/merges.txt
```

The checkpoint is about 4.27 GB and must never be committed. To keep it outside
the repository, set an absolute or repository-relative path:

```bash
SD_MODEL_PATH=/absolute/path/v1-5-pruned-emaonly.ckpt python app.py
```

`SD_VOCAB_PATH` and `SD_MERGES_PATH` can override the tokenizer paths in the
same way. Missing files are reported in the UI with their expected locations.

Security note: `sd/model_converter.py` uses
`torch.load(..., weights_only=False)`. Only load a checkpoint from a source you
trust.

## Run the App

From the repository root:

```bash
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`. The
checkpoint is loaded once when the first valid generation is requested. A
successful result is displayed in the UI and saved as a collision-resistant PNG
under `outputs/`. Text-to-image filenames begin with `txt2img_`; image-to-image
filenames begin with `roommorph_`.

Image-to-image defaults are intentionally lightweight:

```text
resolution:       256 x 256
inference steps:  10
strength:         0.60
CFG:              disabled
CFG scale:        7.5 when enabled
seed:             42
sampler:          DDPM
```

CFG commonly improves prompt adherence but roughly doubles the U-Net batch for
this pipeline, increasing compute and memory use.

Text-to-image defaults to 256 x 256, 10 inference steps, CFG scale 7.5, and seed
42. Its metadata includes the effective style-expanded prompt, negative prompt,
runtime device, parameters, elapsed time, and saved filename.

## Smoke-Test Mode

Construct and launch the complete UI without loading or requiring the
checkpoint:

```bash
ROOMMORPH_SKIP_MODEL_LOAD=1 python app.py
```

The interface clearly marks inference as disabled. Both Generate actions
validate their inputs and settings, then return a disabled-mode message without
fabricating an AI result.

For a non-blocking import check:

```bash
ROOMMORPH_SKIP_MODEL_LOAD=1 python -c "import app; print(type(app.demo).__name__)"
```

## Device and Hardware Notes

Runtime selection is CUDA first, then Apple MPS, then CPU. CPU is always the
idle/offload device, including CPU-only execution.

- Apple Silicon: MPS is used when the installed PyTorch build supports it.
- NVIDIA: CUDA is used when available.
- CPU: supported, but a single image can take several minutes or longer.
- MacBook Air M3 with 8 GB: begin at 256, 10 steps, CFG off.
- Hugging Face CPU Basic: use the same conservative defaults and a queue limit
  of one; loading and generation can still be slow.
- 512 resolution, more steps, and CFG can cause high memory pressure or a much
  longer run.

## Notebook Demo

The original educational notebook remains available at `sd/demo.ipynb` and
supports text-to-image and image-to-image experiments. Start Jupyter from the
repository root:

```bash
jupyter notebook sd/demo.ipynb
```

The notebook uses the same files under `data/`. If you change its resolution,
update `WIDTH`, `HEIGHT`, `LATENTS_WIDTH`, and `LATENTS_HEIGHT` together.

## Example Gallery

Curated before/after portfolio examples will be added later under `assets/`.
See `assets/README.md` for privacy and naming guidance. Ordinary experiments
belong in `outputs/` and remain untracked.

## Known Limitations

- The output is a visual concept, not a construction plan or photorealistic
  guarantee.
- Exact wall geometry, windows, doors, and furniture placement may change.
- Image-to-image influences the full image; it is not inpainting and cannot
  target one masked region.
- The model works on a square center crop, so content near wide-image edges may
  be removed.
- Batch size is one and the only sampler is DDPM.
- Stable Diffusion 1.5 can inherit biases and limitations from its training data.

## Troubleshooting

### `No module named 'transformers'` or `No module named 'gradio'`

Install dependencies with the exact interpreter used to launch the app:

```bash
python -m pip install -r requirements.txt
```

### ``vocab` and `merges` must both be from memory or both filenames`

Both tokenizer files must exist and be passed as filenames. The app supports the
constructor names used by current Transformers 4.x and 5.x releases.

### `No module named 'pytorch_lightning'`

Install `pytorch-lightning` in the active environment. It is included in
`requirements.txt` because legacy checkpoint deserialization can need it.

### Generation runs out of memory

Use 256 resolution, disable CFG, reduce inference steps, close other memory-heavy
applications, and restart the app after a failed run if memory is not released.

### CPU generation looks stuck

Watch the terminal progress bar. The custom implementation is computation-heavy
on CPU, and free CPU hardware may need several minutes for one small image.

## License and Model Attribution

The pretrained model is Stable Diffusion 1.5 and remains subject to the license
and usage terms supplied by its original model provider. Keep attribution and
review those terms before publishing a hosted demo or generated portfolio work.
Model files are not distributed by this repository and must not be committed.

This repository currently has no root `LICENSE` file. Add an explicit code
license before redistributing the project or accepting external contributions;
the model-weight license is separate from the code license.
