import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
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
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

MAX_SEED = np.iinfo(np.int32).max

ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles"
    },
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime"
    },
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2"
    },
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration"
    },
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k"
    },
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer"
    },
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone"
    },
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real"
    },
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles"
    },
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo"
    },
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything"
    },
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight"
    },
    "Hyper-Realistic-Portrait": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
       "weights": "HRP_20.safetensors",
       "adapter_name": "hyper-realistic-portrait"
   },     
    "Ultra-Realistic-Portrait": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
       "weights": "URP_20.safetensors",
       "adapter_name": "ultra-realistic-portrait"
   },     
    "Pixar-Inspired-3D": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
       "weights": "PI3_20.safetensors",
       "adapter_name": "pi3"
   },
    "Noir-Comic-Book": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
       "weights": "Noir-Comic-Book-Panel_20.safetensors",
       "adapter_name": "ncb"
   },  
    "Any-light": {
       "repo": "lilylilith/QIE-2511-MP-AnyLight",
       "weights": "QIE-2511-AnyLight_.safetensors",
       "adapter_name": "any-light"
   }, 
    "Studio-DeLight": {
       "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
       "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
       "adapter_name": "studio-delight"
   }, 
}

LOADED_ADAPTERS = set()

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

@spaces.GPU
def infer(
    images,
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    gc.collect()
    torch.cuda.empty_cache()

    if not images:
        raise gr.Error("Please upload at least one image to edit.")

    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item, tuple) or isinstance(item, list):
                    path_or_img = item[0]
                else:
                    path_or_img = item

                if isinstance(path_or_img, str):
                    pil_images.append(Image.open(path_or_img).convert("RGB"))
                elif isinstance(path_or_img, Image.Image):
                    pil_images.append(path_or_img.convert("RGB"))
                else:
                    pil_images.append(Image.open(path_or_img.name).convert("RGB"))
            except Exception as e:
                print(f"Skipping invalid image item: {e}")
                continue

    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    adapter_name = spec["adapter_name"]

    if adapter_name not in LOADED_ADAPTERS:
        print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
        try:
            pipe.load_lora_weights(
                spec["repo"], 
                weight_name=spec["weights"], 
                adapter_name=adapter_name
            )
            LOADED_ADAPTERS.add(adapter_name)
        except Exception as e:
            raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
    else:
        print(f"--- Adapter {lora_adapter} is already loaded. ---")

    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

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
        
        return result_image, seed

    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()

@spaces.GPU
def infer_example(images, prompt, lora_adapter):
    if not images:
        return None, 0
    
    if isinstance(images, str):
        images_list = [images]
    else:
        images_list = images
        
    result, seed = infer(
        images=images_list,
        prompt=prompt,
        lora_adapter=lora_adapter,
        seed=0,
        randomize_seed=True,
        guidance_scale=1.0,
        steps=4
    )
    return result, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
#main-title h1 {font-size: 2.3em !important;}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-Image-Edit-2511-LoRAs-Fast**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized [LoRA](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image-Edit-2511) adapters. Open on [GitHub](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load).")

        with gr.Row(equal_height=True):
            with gr.Column():
                images = gr.Gallery(
                    label="Upload Images", 
                    type="filepath", 
                    columns=2, 
                    rows=1, 
                    height=300,
                    allow_preview=True
                )
                
                prompt = gr.Text(
                    label="Edit Prompt",
                    #max_lines=1,
                    show_label=True,
                    placeholder="e.g., transform into anime..",
                )

                run_button = gr.Button("Edit Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=365)
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=list(ADAPTER_SPECS.keys()),
                        value="Photo-to-Anime"
                    )
                
                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)
        
        gr.Examples(
            examples=[
                [["examples/B.jpg"], "Transform into anime.", "Photo-to-Anime"],
                [["examples/HRP.jpg"], "Transform into a hyper-realistic face portrait.", "Hyper-Realistic-Portrait"],
                [["examples/A.jpeg"], "Rotate the camera 45 degrees to the right.", "Multiple-Angles"],
                [["examples/U.jpg"], "Upscale this picture to 4K resolution.", "Upscaler"],
                [["examples/L1.jpg", "examples/L2.jpg"], "Apply the lighting from image 2 to image 1.", "Any-light"],
                [["examples/PP1.jpg"], "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by hf‪‪‬ preserving realistic texture and details", "Polaroid-Photo"],
                [["examples/Z1.jpg"], "Front-right quarter view.", "Fal-Multiple-Angles"],
                [["examples/SL.jpg"], "Neutral uniform lighting Preserve identity and composition.", "Studio-DeLight"],
                [["examples/PI.jpg"], "Transform it into Pixar-inspired 3D.", "Pixar-Inspired-3D"],
                [["examples/MT.jpg"], "Paint with manga tone.", "Manga-Tone"],
                [["examples/NCB.jpg"], "Transform into a noir comic book style.", "Noir-Comic-Book"],
                [["examples/URP.jpg"], "ultra-realistic portrait.", "Ultra-Realistic-Portrait"],
                [["examples/MN.jpg"], "Transform into Midnight Noir Eyes Spotlight.", "Midnight-Noir-Eyes-Spotlight"],
                [["examples/ST1.jpg", "examples/ST2.jpg"], "Convert Image 1 to the style of Image 2.", "Style-Transfer"],
                [["examples/R1.jpg"], "Change the picture to realistic photograph.", "Anything2Real"],
                [["examples/UA.jpeg"], "Unblur and upscale.", "Unblur-Anything"],
                [["examples/L1.jpg", "examples/L2.jpg"], "Refer to the color tone, remove the original lighting from Image 1, and relight Image 1 based on the lighting and color tone of Image 2.", "Light-Migration"],
                [["examples/P1.jpg"], "Transform into anime (while preserving the background and remaining elements maintaining realism and original details.)", "Anime-V2"],
            ],
            inputs=[images, prompt, lora_adapter],
            outputs=[output_image, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )
        
        gr.Markdown("[*](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)This is still an experimental Space for Qwen-Image-Edit-2511.")

    run_button.click(
        fn=infer,
        inputs=[images, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)
