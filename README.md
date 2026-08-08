# **[Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)**

Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load is an experimental, high-performance image editing and style-transfer platform built on top of the `Qwen/Qwen-Image-Edit-2511` base model and an optimized transformer architecture (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`). The application integrates Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`) to achieve low VRAM footprints and accelerated 4-step image manipulation.

Using a **Lazy Loading** design for LoRA adapters, the system dynamically downloads and fuses task-specific adapters on demand—including Multiple Angles, Photo-to-Anime, Anime-V2, Light Migration, Upscaler, Style Transfer, Manga Tone, Anything2Real, Polaroid Photo, Unblur Anything, Midnight Noir, Hyper-Realistic Portrait, Ultra-Realistic Portrait, Pixar-Inspired 3D, Noir Comic Book, Any Light, Studio DeLight, and Cinematic FlatLog. The web workspace is served via a custom, single-page web app built with a FastAPI backend server (`gradio.Server`) and a dark-mode frontend interface featuring a dual-view canvas, A/B comparison slider, history filmstrip, and interactive prompt suggestions.

<img width="1919" height="847" alt="image (1)" src="https://github.com/user-attachments/assets/16525234-b2c6-4dc4-afba-b414872e7e60" />

### **Key Features**

* **Lazy-Loaded Adapter Registry:** On-demand downloading and weight-fusing for 19+ specialized LoRA adapters (e.g., *Multiple-Angles*, *Photo-to-Anime*, *Anime-V2*, *Light-Migration*, *Upscaler*, *Style-Transfer*, *Manga-Tone*, *Anything2Real*, *Polaroid-Photo*, *Unblur-Anything*, *Pixar-Inspired-3D*, *Studio-DeLight*, and *Cinematic-FlatLog*).
* **Flash Attention 3 (FA3) Acceleration:** Hooks natively into the `QwenDoubleStreamAttnProcessorFA3` processor layer to accelerate cross-attention inference phases while reducing active GPU memory consumption.
* **Text-Guided Image Editing:** Offers camera angle rotations, shadow removal, uniform studio relighting, skin detail refinement, scene propagation, and 4K upscaling.
* **Studio SPA Interface:** An interactive single-page application built with modern vanilla web components—featuring an A/B image comparison slider, history filmstrip, quick prompt chips, and drag-and-drop file support.
* **Smart Aspect Ratio Snapping:** Automatically resizes uploaded images to stay within 1024px while snapping width and height to multiples of 8 to prevent shape mismatch errors during inference.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── A.jpeg
│   ├── B.jpg
│   ├── CFL.jpg
│   ├── HRP.jpg
│   ├── HS1.jpg
│   ├── HS2.jpg
│   ├── L1.jpg
│   ├── L2.jpg
│   ├── MN.jpg
│   ├── MT.jpg
│   ├── NCB.jpg
│   ├── P1.jpg
│   ├── P2.jpg
│   ├── PI.jpg
│   ├── PP1.jpg
│   ├── R1.jpg
│   ├── SL.jpg
│   ├── ST1.jpg
│   ├── ST2.jpg
│   ├── U.jpg
│   ├── UA.jpeg
│   ├── URP.jpg
│   ├── Z1.jpg
│   ├── Z2.jpg
│   └── Z3.jpg
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py
├── index.html
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.12** is needed; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for better compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git
cd Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.22.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
kernels==0.16.0
```

### **Usage**

Once the web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Asset:** Drag and drop an image into the main canvas workspace, paste an image from your clipboard, or click the upload icon in the left rail.
2. **Select Style / LoRA:** Choose your target editing task from the **Style / LoRA** dropdown menu in the right inspector panel. The adapter weights will download lazily on first use.
3. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to instantly fill it. Press ⌘/Ctrl + Enter or click **Edit Image**.
4. **Compare & Chain:** Use the **Compare** tool on the left rail to view an A/B slider of the before and after states. Click **Use as Input** to chain multiple edits sequentially.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load/blob/main/LICENSE)
* Thanks to **AK** (aka [akhaliq](https://huggingface.co/akhaliq)) for the initial [PR #27](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast/discussions/27), customization, and motivation.
