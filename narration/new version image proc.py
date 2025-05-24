"""from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import openai

# 🔧 OpenAI API anahtarını ayarla
openai.api_key = "token"

# 📸 Görsel altyapı modeli (Görselden cümle üretimi)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 📷 Görseli aç
image = Image.open("profesyonel-fotograf-cekimi-1.jpg").convert("RGB")

# 🎛️ Görseli modele uygun hale getir
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# 📝 Açıklama üret
output_ids = model.generate(pixel_values, max_length=16)

caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Fotoğraf açıklaması:", caption)


# 🧠 Hikaye üretme fonksiyonu
def hikaye_olustur(kullanici_cumlesi):
    prompt = f"'{kullanici_cumlesi}' cümlesine dayalı yaratıcı ve kısa bir hikaye oluştur."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.8
        )

        hikaye = response.choices[0].message["content"]
        return hikaye

    except Exception as e:
        return f"Hata oluştu: {e}"


# 🎬 Hikaye üret
sonuc = hikaye_olustur(caption)

print("\n--- Oluşturulan Hikaye ---")
print(sonuc)
"""
"""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import requests
import json

# 📸 Görsel altyapı modeli (Görselden cümle üretimi)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 📷 Görseli aç
image = Image.open("profesyonel-fotograf-cekimi-1.jpg").convert("RGB")

# 🎛️ Görseli modele uygun hale getir
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# 📝 Açıklama üret
output_ids = model.generate(pixel_values, max_length=16)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Fotoğraf açıklaması:", caption)

# 🧠 DeepSeek API ayarları
DEESEEK_API_KEY = "token"
DEESEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 🧠 Hikaye üretme fonksiyonu
def hikaye_olustur(cumle):
    prompt = f"'{cumle}' cümlesine dayalı yaratıcı ve kısa bir hikaye oluştur."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEESEEK_API_KEY}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 400
    }

    try:
        response = requests.post(DEESEEK_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        hikaye = response.json()["choices"][0]["message"]["content"]
        return hikaye

    except Exception as e:
        return f"Hata oluştu: {e}"

# 🎬 Hikaye üret
sonuc = hikaye_olustur(caption)
print("\n--- Oluşturulan Hikaye ---")
print(sonuc)
"""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image
import torch

# Cihaz ayarı (GPU varsa kullan)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Görselden açıklama üretme modeli yükle
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)

# Görseli aç
image = Image.open("wp2123042.jpg").convert("RGB")

# Görseli modele uygun hale getir
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Açıklama üret
output_ids = model.generate(pixel_values, max_length=16)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Fotoğraf açıklaması:", caption)


# 2) GPT-2 ile hikaye oluşturma (text-generation pipeline)
story_generator = pipeline('text-generation', model='gpt2', device=0 if device=="cuda" else -1)

# Prompt oluştur (caption'a dayalı kısa hikaye)
prompt = f"creative story based on this sentence: '{caption}'"

# Hikaye üret
stories = story_generator(prompt, max_length=500, num_return_sequences=1)

print("\n--- Oluşturulan Hikaye ---")
print(stories[0]['generated_text'])
