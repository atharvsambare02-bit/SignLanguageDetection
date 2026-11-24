import cv2
from pathlib import Path

SAVE_DIR = Path(
    "Tensorflow/workspace/images/collectedimages/no"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

image_number = 21
target_images = 20

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nNO gesture image collection started!")
print("Show THUMBS-DOWN.")
print("Press S to save an image.")
print("Press Q to quit.\n")

while image_number <= 40:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"NO Images: {image_number - 21}/{target_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Show THUMBS-DOWN | Press S to save | Q to quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect NO Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        filename = f"no_{image_number:03d}.jpg"
        filepath = SAVE_DIR / filename

        success = cv2.imwrite(str(filepath), frame)

        if success:
            print(f"Saved: {filename}")
            image_number += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nImage collection finished.")
print(f"Images collected: {image_number - 21}")
print(f"Saved in: {SAVE_DIR}")
