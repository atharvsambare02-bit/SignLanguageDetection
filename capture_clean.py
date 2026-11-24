import cv2
import sys
from pathlib import Path

if len(sys.argv) != 2 or sys.argv[1] not in ["yes", "no"]:
    print("Usage: python capture_clean.py yes")
    print("   or: python capture_clean.py no")
    exit()

gesture = sys.argv[1].lower()

SAVE_DIR = Path(f"Tensorflow/workspace/images/collectedimages/{gesture}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

image_number = 21
target_images = 20

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print(f"\nCollecting {gesture.upper()} images")
print("Show the correct gesture.")
print("Press S to save | Q to quit")

while image_number <= 40:
    ret, frame = cap.read()

    if not ret:
        break

    clean_frame = cv2.flip(frame, 1)
    display_frame = clean_frame.copy()

    cv2.putText(
        display_frame,
        f"{gesture.upper()} Images: {image_number - 21}/{target_images}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )

    cv2.putText(
        display_frame,
        "Press S to save | Q to quit",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )

    cv2.imshow(f"Collect {gesture.upper()} Images", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = f"{gesture}_{image_number:03d}.jpg"
        filepath = SAVE_DIR / filename

        # Save the clean frame without instruction text
        if cv2.imwrite(str(filepath), clean_frame):
            print(f"Saved: {filename}")
            image_number += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinished. Saved {image_number - 21} images.")
print(f"Location: {SAVE_DIR}")
