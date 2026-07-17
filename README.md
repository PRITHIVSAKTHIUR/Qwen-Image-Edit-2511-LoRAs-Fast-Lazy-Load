# **[Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)**

Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load is an experimental, high-performance image editing and style-transfer platform built upon the `Qwen/Qwen-Image-Edit-2511` base pipeline and an optimized transformer backbone (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`). The suite implements Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`) to achieve low VRAM consumption and rapid 4-step image manipulation.

Featuring a **Lazy Loading** architecture for LoRA adapters, the application dynamically fetches and fuses adapters on demand—including Multiple Angles, Photo-to-Anime, Light Migration, Upscaler, and Pixar 3D—only when requested. The backend is coupled with a custom dark-themed Gradio web application with drag-and-drop reference galleries, interactive prompt chips, and instant example loading.

<img width="1714" height="1596" alt="image" src="https://github.com/user-attachments/assets/7c665c83-d5a0-492a-9ede-074382c6c46a" />

### **Key Features**

* **Lazy-Loaded Adapter Hub:** On-demand downloading and weight-fusing for 19+ pre-configured specialized LoRA adapters (e.g., *Multiple-Angles*, *Photo-to-Anime*, *Light-Migration*, *Upscaler*, *Style-Transfer*, *Pixar-Inspired-3D*, *Studio-DeLight*, and more).
* **Flash Attention 3 (FA3) Acceleration:** Hooks into `QwenDoubleStreamAttnProcessorFA3` to accelerate attention processing and reduce memory overhead during multi-image cross-attention passes.
* **Multi-Image Reference Manipulation:** Supports uploading multiple reference images to drive guided edits (e.g., migrating lighting from Image 2 onto Image 1 or applying style transfers across subjects).
* **Custom Headless-Style UI:** Features a dark-mode terminal layout with client-side JavaScript gallery management, drag-and-drop uploader zones, live toast feedback, and quick suggestion chips.
* **Automatic Dimension Snapping:** Calculates source image aspect ratios and snaps target dimensions to multiples of 8 (bounded by a 1024px maximum edge) to prevent tensor shape errors during diffusion.

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
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To set up the Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load environment locally, configure your environment according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.12** is required; Python **3.14** is recommended for best performance.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal system compatibility and Flash Attention 3 execution.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid environment setup and exact dependency synchronization based on `uv.lock`.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh`
* **Windows:** `powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"`

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

---

#### **Standard PIP Installation**

**1. Update Package Manager**
Upgrade your local pip installer:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer builds, and web client libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt

```

### **Usage**

Once the FastAPI web client launches, open your browser to the local address output in your terminal (typically `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`).

1. **Upload Images:** Drop one or more images into the upload area.
* For single-subject edits (e.g., *"Transform into anime"*), upload 1 image.
* For reference-guided edits (e.g., *"Apply the lighting from image 2 to image 1"*), upload the target image first followed by the reference style/lighting image.


2. **Select Style / LoRA:** Choose the desired editing adapter from the **Editing Style / LoRA** dropdown menu. The adapter weights will be downloaded lazily and fused on first execution.
3. **Set Prompt:** Type your edit instructions or click one of the **Quick Prompts** chips to auto-fill a directive.
4. **Execute:** Click **Edit Image**. The web interface will show an active loader bar while the GPU processes the diffusion steps, returning the final image upon completion.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load/blob/main/LICENSE)
