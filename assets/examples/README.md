# Curated Examples

These files are selected portfolio examples, not an archive of ordinary runs.
The application continues to save all successful generations to the ignored
`outputs/` directory.

## Image-to-Image

- Input: `image-to-image/original-bedroom.jpg` (2500 x 1667).
- Output: `image-to-image/industrial-bedroom-concept.png` (512 x 512 RGB).
- Historical prompt and timing metadata were not retained for this early run,
  so it is presented only as a qualitative example.

## Text-to-Image

- Output: `text-to-image/japandi-bedroom-concept.png` (512 x 512 RGB).
- Prompt: `text-to-image/prompt.txt`.
- Runtime: Apple MPS on a MacBook Air with Apple M3 and 8 GB unified memory.
- Parameters: 50 inference steps, CFG scale 8.0, seed 42, DDPM sampler.
- Cold model load: approximately 68.7 seconds.
- Pipeline generation time: 2545.75 seconds (about 42 minutes 26 seconds).

The timing is one observed local run from 2026-08-25, not a generalized
benchmark. It excludes the cold model load and should not be extrapolated to
other devices or settings.
