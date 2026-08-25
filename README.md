# Stable Diffusion Demo

Implementasi Stable Diffusion sederhana dengan PyTorch. Repo ini memuat komponen inti model di folder `sd/`, loader untuk checkpoint Stable Diffusion v1.5, serta notebook demo untuk text-to-image dan image-to-image.

## Struktur Project

```text
.
├── data/
│   ├── v1-5-pruned-emaonly.ckpt
│   ├── vocab.json
│   └── merges.txt
├── images/
│   └── dog.png
├── outputs/
├── sd/
│   ├── demo.ipynb
│   ├── pipeline.py
│   ├── model_loader.py
│   ├── model_converter.py
│   ├── clip.py
│   ├── diffusion.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── ddpm.py
│   └── attention.py
└── README.md
```

Folder `data/`, `images/`, dan `outputs/` di-ignore oleh git karena berisi file besar, input lokal, dan hasil generate.

## Dependency

Gunakan Python/Jupyter kernel yang sama saat install dan menjalankan notebook. Di setup ini Jupyter memakai Anaconda `base`:

```bash
/opt/anaconda3/bin/python -m pip install torch numpy pillow tqdm transformers pytorch-lightning jupyter ipykernel
```

Kalau sudah punya sebagian package, cukup install yang hilang:

```bash
/opt/anaconda3/bin/python -m pip install transformers pytorch-lightning
```

Package penting:

- `torch`: menjalankan model.
- `transformers`: memuat `CLIPTokenizer`.
- `pytorch-lightning`: dibutuhkan saat `torch.load` membaca checkpoint `.ckpt` lama.
- `pillow`: membaca dan menyimpan gambar.
- `tqdm`: progress bar inference.

## File Model Yang Wajib Ada

Pastikan file berikut ada di folder `data/`:

```text
data/v1-5-pruned-emaonly.ckpt
data/vocab.json
data/merges.txt
```

Notebook akan gagal kalau salah satu file itu tidak ada.

Catatan keamanan: `model_converter.py` memakai `torch.load(..., weights_only=False)` untuk membaca checkpoint. Jalankan hanya checkpoint yang kamu percaya.

## Cara Menjalankan Demo

Dari root project:

```bash
cd "/Users/acit/Documents/MyWork/Stabel Diffusion"
/opt/anaconda3/bin/jupyter notebook sd/demo.ipynb
```

Di notebook:

1. Jalankan cell pertama untuk setup path, device, tokenizer, dan load model.
2. Jalankan cell `Image to Image` untuk memakai `images/dog.png` sebagai input.
3. Jalankan cell `TEXT-TO-IMAGE` untuk generate dari prompt saja.

Output akan disimpan ke folder `outputs/` dengan nama bertimestamp, misalnya:

```text
outputs/img2img_20260825_140805.png
outputs/txt2img_20260825_141000.png
```

Hasil juga ditampilkan langsung di notebook dengan `display(result)`.

## Pengaturan Performa

Notebook saat ini memakai:

```python
pipeline.WIDTH = 512
pipeline.HEIGHT = 512
```

Untuk MacBook Air atau RAM terbatas, kalau lambat atau memory penuh, turunkan:

```python
pipeline.WIDTH = 256
pipeline.HEIGHT = 256
pipeline.LATENTS_WIDTH = pipeline.WIDTH // 8
pipeline.LATENTS_HEIGHT = pipeline.HEIGHT // 8
```

Parameter yang paling terasa:

- `n_inference_steps`: makin besar, makin lama.
- `do_cfg=True`: hasil biasanya lebih mengikuti prompt, tapi lebih berat.
- `do_cfg=False`: lebih ringan untuk demo cepat.
- `strength` pada image-to-image: makin tinggi, hasil makin jauh dari input.

## Troubleshooting

### `No module named 'transformers'`

Package `transformers` belum terinstall di kernel Jupyter yang aktif.

```bash
/opt/anaconda3/bin/python -m pip install transformers
```

### ``vocab` and `merges` must be both be from memory or both filenames`

Versi `transformers` baru memakai parameter:

```python
CLIPTokenizer(vocab=..., merges=...)
```

Bukan:

```python
CLIPTokenizer(vocab_file=..., merges_file=...)
```

Notebook sudah memakai format baru.

### `No module named 'pytorch_lightning'`

Checkpoint `.ckpt` membutuhkan PyTorch Lightning saat dibaca oleh `torch.load`.

```bash
/opt/anaconda3/bin/python -m pip install pytorch-lightning
```

### `CLIPTokenizer has no attribute batch_encode_plus`

API tokenizer lama sudah tidak tersedia di `transformers` baru. `sd/pipeline.py` sudah diperbarui agar memakai:

```python
tokenizer(...).input_ids
```

### Notebook terlihat masih loading setelah progress 100%

Pastikan notebook memakai `display(result)`, bukan `result.show()`. `result.show()` mencoba membuka aplikasi gambar eksternal di macOS dan bisa membuat cell terlihat menggantung.

### Output seperti file lama

Versi lama menyimpan semua hasil ke `outputs/demo_output.png`, sehingga file lama bisa tertimpa. Notebook sekarang memakai nama bertimestamp agar setiap run punya file baru.

### Data tiba-tiba hilang

Pastikan file model dan tokenizer masih ada di `data/`. Kalau folder sempat berubah nama atau file berpindah, cari ulang:

```bash
find /Users/acit/Documents /Users/acit/Downloads -name 'v1-5-pruned-emaonly.ckpt' -o -name 'vocab.json' -o -name 'merges.txt'
```

## Catatan

Repo ini fokus untuk belajar cara kerja Stable Diffusion, bukan wrapper production. Implementasi saat ini:

- batch size 1.
- sampler `ddpm`.
- text-to-image dan image-to-image.
- checkpoint format Stable Diffusion v1.5 standard `.ckpt`.
