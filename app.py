import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
import json
from io import BytesIO
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
    },
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime",
    },
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2",
    },
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration",
    },
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k",
    },
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer",
    },
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone",
    },
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real",
    },
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles",
    },
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo",
    },
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything",
    },
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight",
    },
    "Hyper-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "hyper-realistic-portrait",
    },
    "Ultra-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "ultra-realistic-portrait",
    },
    "Pixar-Inspired-3D": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
        "weights": "PI3_20.safetensors",
        "adapter_name": "pi3",
    },
    "Noir-Comic-Book": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
        "weights": "Noir-Comic-Book-Panel_20.safetensors",
        "adapter_name": "ncb",
    },
    "Any-light": {
        "repo": "lilylilith/QIE-2511-MP-AnyLight",
        "weights": "QIE-2511-AnyLight_.safetensors",
        "adapter_name": "any-light",
    },
    "Studio-DeLight": {
        "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
        "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
        "adapter_name": "studio-delight",
    },
    "Cinematic-FlatLog": {
        "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
        "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
        "adapter_name": "flat-log",
    },
}

LOADED_ADAPTERS: set = set()
ADAPTER_NAMES = list(ADAPTER_SPECS.keys())

EXAMPLES_CONFIG = [
    {"images": ["examples/B.jpg"],                          "prompt": "Transform into anime.",                                                                                           "lora": "Photo-to-Anime"},
    {"images": ["examples/HRP.jpg"],                        "prompt": "Transform into a hyper-realistic face portrait.",                                                                 "lora": "Hyper-Realistic-Portrait"},
    {"images": ["examples/A.jpeg"],                         "prompt": "Rotate the camera 45 degrees to the right.",                                                                      "lora": "Multiple-Angles"},
    {"images": ["examples/U.jpg"],                          "prompt": "Upscale this picture to 4K resolution.",                                                                          "lora": "Upscaler"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Apply the lighting from image 2 to image 1.",                                                                     "lora": "Any-light"},
    {"images": ["examples/PP1.jpg"],                        "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed preserving realistic texture and details.", "lora": "Polaroid-Photo"},
    {"images": ["examples/Z1.jpg"],                         "prompt": "Front-right quarter view.",                                                                                       "lora": "Fal-Multiple-Angles"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Transform into a cinematic flat log.",                                                                            "lora": "Cinematic-FlatLog"},
    {"images": ["examples/SL.jpg"],                         "prompt": "Neutral uniform lighting. Preserve identity and composition.",                                                    "lora": "Studio-DeLight"},
    {"images": ["examples/PI.jpg"],                         "prompt": "Transform it into Pixar-inspired 3D.",                                                                            "lora": "Pixar-Inspired-3D"},
    {"images": ["examples/MT.jpg"],                         "prompt": "Paint with manga tone.",                                                                                          "lora": "Manga-Tone"},
    {"images": ["examples/NCB.jpg"],                        "prompt": "Transform into a noir comic book style.",                                                                         "lora": "Noir-Comic-Book"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Ultra-realistic portrait.",                                                                                       "lora": "Ultra-Realistic-Portrait"},
    {"images": ["examples/MN.jpg"],                         "prompt": "Transform into Midnight Noir Eyes Spotlight.",                                                                    "lora": "Midnight-Noir-Eyes-Spotlight"},
    {"images": ["examples/ST1.jpg", "examples/ST2.jpg"],    "prompt": "Convert Image 1 to the style of Image 2.",                                                                        "lora": "Style-Transfer"},
    {"images": ["examples/R1.jpg"],                         "prompt": "Change the picture to realistic photograph.",                                                                     "lora": "Anything2Real"},
    {"images": ["examples/UA.jpeg"],                        "prompt": "Unblur and upscale.",                                                                                             "lora": "Unblur-Anything"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Refer to the color tone, remove the original lighting from Image 1, and relight Image 1 based on the lighting and color tone of Image 2.", "lora": "Light-Migration"},
    {"images": ["examples/P1.jpg"],                         "prompt": "Transform into anime (while preserving the background and remaining elements maintaining realism and original details.)", "lora": "Anime-V2"},
]


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def build_client_config():
    """Static config consumed by the frontend: LoRA list + example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "lora": ex["lora"],
            "prompt": ex["prompt"],
        })
    return {
        "loras": ADAPTER_NAMES,
        "default_lora": "Photo-to-Anime",
        "examples": examples,
    }


print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples and {len(ADAPTER_NAMES)} LoRAs.")


def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images


def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw = 1024
        nh = int(nw * h / w)
    else:
        nh = 1024
        nw = int(nh * w / h)
    return (nw // 8) * 8, (nh // 8) * 8


# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Qwen-Image-Edit-2511-LoRAs-Fast")


@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    lora_adapter: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Edit one or more images with Qwen-Image-Edit-2511 + a lazily-loaded LoRA.

    Returns {"image": <base64 PNG data URL>, "seed": <seed used>}.
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    adapter_name = spec["adapter_name"]
    if adapter_name not in LOADED_ADAPTERS:
        print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
        try:
            pipe.load_lora_weights(spec["repo"], weight_name=spec["weights"], adapter_name=adapter_name)
            LOADED_ADAPTERS.add(adapter_name)
        except Exception as e:
            raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
    else:
        print(f"--- Adapter {lora_adapter} already loaded. ---")

    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )
    width, height = update_dimensions_on_upload(pil_images[0])

    try:
        result_image = pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt + LoRA for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "lora": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "lora": ex["lora"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: LoRA choices + example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)
