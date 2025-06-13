from diffusers import StableDiffusionImg2ImgPipeline
import torch
import gradio as gr
from PIL import Image

# Modeli yükle
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

def convert_to_year_style(image, year):
    prompt = f"A photo of this place in the year {year}, old film style, realistic, grainy"
    image = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
    return image

# Arayüz
demo = gr.Interface(
    fn=convert_to_year_style,
    inputs=[
        gr.Image(type="pil", label="Bir Görsel Yükleyin"),
        gr.Textbox(label="Hedef Yıl (örneğin: 1980)")
    ],
    outputs="image",
    title="Zaman Yolculuğu Görseli",
    description="Görseli yükleyin ve hedef yılı girin. O yıla uygun görsel estetiğini otomatik üretir."
)

demo.launch()

