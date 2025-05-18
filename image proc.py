from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# Model ve işlemci yükle
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Fotoğrafı aç
image = Image.open("profesyonel-fotograf-cekimi-1.jpg")

# Görüntüyü işlemci ile hazırla
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Metin tahmini yap0
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Fotoğraf açıklaması:", caption)


