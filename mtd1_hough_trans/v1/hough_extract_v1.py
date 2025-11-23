import cv2
import numpy as np
import json

IMAGE_FILE = "mrt_clean.png"

# ----------------------------
# Downtown Line HSV Range (Blue)
# ----------------------------
DOWNTOWN_LINE_RANGE = {
    "lower": (95, 70, 70),
    "upper": (130, 255, 255)
}

# ----------------------------

def save_segments(img, segments):
    h, w = img.shape[:2]

    output = {
        "line": "downtown_line",
        "image_size": {"width": w, "height": h},
        "segments": segments
    }

    with open("dtl_v1.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved downtown_line.json with {len(segments)} segments.")


def detect_lines(mask):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    lines = cv2.HoughLinesP(
        cleaned,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=20,
        maxLineGap=15
    )

    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(
                {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            )

    return segments


def extract_downtown_line():
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        raise FileNotFoundError("Image not found at " + IMAGE_FILE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        DOWNTOWN_LINE_RANGE["lower"],
        DOWNTOWN_LINE_RANGE["upper"]
    )

    segments = detect_lines(mask)

    save_segments(img, segments)


if __name__ == "__main__":
    extract_downtown_line()
