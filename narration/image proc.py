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
"""
import openai

# OpenAI API anahtarını buraya gir
client = openai.OpenAI(api_key = "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")

def hikaye_olustur(kullanici_cumlesi):
    prompt = f"'{kullanici_cumlesi}' Create a creative and short story based on the sentence."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Eğer gpt-4 yoksa bu kullanılabilir
            messages=[
                {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.8
        )

        hikaye = response.choices[0].message.content
        return hikaye

    except Exception as e:
        return f"Hata oluştu: {e}"

# Kullanıcıdan cümle al
sonuc = hikaye_olustur(caption)

print("\n--- Created Story ---")
print(sonuc)
"""

"""
import openai

# DeepSeek API anahtarını buraya yaz
api_key = "sk-e3739b11f4e14d7aa60b9b022f0e67a4"

# DeepSeek API URL’si
base_url = "https://api.deepseek.com/v1"

# OpenAI client gibi kullanılır ama DeepSeek'e yönlendirilir
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)

def hikaye_olustur(cumle):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek’in sohbet modeli
            messages=[
                {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
                {"role": "user", "content": f"Lütfen bu cümleye göre yaratıcı ve kısa bir hikaye yaz: {cumle}"}
            ],
            max_tokens=400,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {e}"

# Kullanıcıdan cümle al

hikaye = hikaye_olustur(caption)

print("\n--- Oluşturulan Hikaye ---")
print(hikaye)
"""
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2-"
API_TOKEN = "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"

headers = {"Authorization": f"Bearer {API_TOKEN}"}


def hikaye_olustur(cumle):
    data = {
        "inputs": f"Write a short story based on this sentence: {cumle}",
        "parameters": {"max_length": 50},
    }
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        # response yapısını kontrol etmek için:
        # print(result)

        # Bazı modellerde yanıt ['generated_text'] altında, bazılarında ise liste halinde olabilir
        if isinstance(result, list) and 'generated_text' in result[0]:
            story = result[0]['generated_text']
        elif 'generated_text' in result:
            story = result['generated_text']
        else:
            story = str(result)
        return story
    else:
        return f"Hata: {response.status_code} - {response.text}"

print("\n--- Oluşan Hikaye ---\n")
print(hikaye_olustur(caption))

