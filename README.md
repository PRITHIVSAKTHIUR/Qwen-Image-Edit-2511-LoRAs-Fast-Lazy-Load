# **Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load**

> A Gradio-based demonstration for the Qwen/Qwen-Image-Edit-2511 model with lazy-loaded LoRA adapters for advanced single- and multi-image editing. Supports 7+ specialized LoRAs including photo-to-anime, multi-angle camera control, pose transfer (Any-Pose), upscaling, style transfer, light migration, and manga tone. Features fast inference (4 steps default) with Flash Attention 3 and dynamic adapter loading to optimize memory.

## Features

- **Multi-Image Support**: Upload one or more images via gallery (e.g., subject + reference for pose/style transfer).
- **Lazy LoRA Loading**: 7 adapters load on-demand only when selected, minimizing VRAM usage.
- **Advanced Editing Tasks**:
  - Photo-to-Anime: Realistic to anime style
  - Multiple-Angles: Camera rotation/view changes
  - Any-Pose: Precise pose transfer from reference
  - Upscaler: High-resolution enhancement
  - Style-Transfer: Apply artistic style from reference
  - Light-Migration: Match lighting/color tone
  - Manga-Tone: Black-and-white manga aesthetic & more.
- **Rapid Inference**: Flash Attention 3 enabled; 4-step default with bfloat16.
- **Auto-Resizing**: Preserves aspect ratio up to 1024px max edge (multiples of 8).
- **Custom Theme**: OrangeRedTheme with clean, responsive layout.
- **Examples**: 7 curated multi/single-image scenarios.
- **Queueing**: Up to 30 concurrent jobs.

<img width="1390" height="787" alt="Screenshot 2026-01-02 at 10-59-26 Qwen-Image-Edit-2511-LoRAs-Fast - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/704e07e5-8928-464e-b50d-906d0c156f88" />

<img width="1422" height="791" alt="Screenshot 2026-01-02 at 10-59-59 Qwen-Image-Edit-2511-LoRAs-Fast - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/1e001022-2c13-4d11-a53f-79beca46ffe9" />

**Note**: This is an experimental Space for the newer Qwen-Image-Edit-2511 model. For stable performance, consider the [2509 version](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2509-LoRAs-Fast).

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (required for bfloat16 and Flash Attention 3).
- Stable internet for initial model/LoRA downloads.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git
   cd Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/diffusers.git
   git+https://github.com/huggingface/peft.git
   transformers==4.57.3
   huggingface_hub
   sentencepiece
   torchvision
   kernels
   spaces
   hf_xet
   gradio #gradio@6
   torch
   numpy
   av
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage

1. **Upload Images**: Use gallery to add one or more images (e.g., person + pose reference).

2. **Select Adapter**: Choose from 7 styles (default: Photo-to-Anime).

3. **Enter Prompt**: Describe the edit (e.g., "Make the person do the exact same pose").

4. **Configure (Optional)**: Expand "Advanced Settings" for seed, guidance, steps.

5. **Edit Image**: Click "Edit Image" to generate output.

### Supported Adapters

| Adapter            | Use Case                                      |
|--------------------|-----------------------------------------------|
| Photo-to-Anime    | Realistic to anime conversion                 |
| Multiple-Angles   | Camera angle/rotation changes                  |
| Any-Pose          | Precise pose transfer from reference          |
| Upscaler          | 2K/4K resolution enhancement                  |
| Style-Transfer    | Apply artistic style from reference           |
| Light-Migration   | Match lighting and color tone                 |
| Manga-Tone        | Black-and-white manga aesthetic               |

## Examples

| Input Images                  | Prompt Example                                                                                             | Adapter             |
|-------------------------------|------------------------------------------------------------------------------------------------------------|---------------------|
| examples/B.jpg                | "Transform into anime."                                                                                   | Photo-to-Anime     |
| examples/A.jpeg               | "Rotate the camera 45 degrees to the right."                                                              | Multiple-Angles    |
| examples/U.jpg                | "Upscale this picture to 4K resolution."                                                                  | Upscaler           |
| examples/MT.jpg               | "Paint with manga tone."                                                                                  | Manga-Tone         |
| examples/ST1.jpg + examples/ST2.jpg | "Convert Image 1 to the style of Image 2."                                                           | Style-Transfer     |
| examples/L1.jpg + examples/L2.jpg | "Relight Image 1 based on the lighting and color tone of Image 2."                                       | Light-Migration    |
| examples/P1.jpg + examples/P2.jpg | "Make the person in image 1 do the exact same pose of the person in image 2."                            | Any-Pose           |

## Troubleshooting

- **Adapter Loading**: First selection downloads LoRA; monitor console.
- **OOM**: Reduce steps/resolution; clear cache with `torch.cuda.empty_cache()`.
- **Flash Attention Fails**: Fallback to default; requires compatible CUDA.
- **Gallery Input**: Supports filepaths, tuples, or PIL objects.
- **No Output**: Ensure at least one valid image and descriptive prompt.

## Contributing

Contributions welcome! Add new adapters to `ADAPTER_SPECS`, improve multi-image handling, or enhance prompts.

Repository: [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
