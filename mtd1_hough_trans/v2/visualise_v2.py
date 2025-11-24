import json
import cv2
import numpy as np

JSON_FILE = "ewl_v2.json"
OUTPUT_IMAGE = "ewl_v2.png"

def main():
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    segments = data["segments"]
    width = data["image_width"]
    height = data["image_height"]

    print(f"Loaded {len(segments)} segments")
    print(f"Canvas size: {width} x {height}")

    # Create white canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw all Downtown Line segments
    for seg in segments:
        x1, y1 = seg["x1"], seg["y1"]
        x2, y2 = seg["x2"], seg["y2"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 175, 0), 3, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"Saved DTL-only image to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
