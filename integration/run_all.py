# run_all.py

from segmentation.segment import segment_image
from generation.generate import generate_past_future
from narration.narrate import create_narrative

def run_pipeline(image_path):
    print("1. Segmenting...")
    segments = segment_image(image_path)

    print("2. Generating past/future...")
    past_img, future_img = generate_past_future(image_path, segments)

    print("3. Creating narrative...")
    narrative = create_narrative(segments, past_img, future_img)

    print("\n🎉 Completed")
    return {
        "original": image_path,
        "segments": segments,
        "past": past_img,
        "future": future_img,
        "narrative": narrative
    }

if __name__ == "__main__":
    result = run_pipeline("assets/example.jpg")
    print(result["narrative"])
