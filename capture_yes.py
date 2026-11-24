import cv2
from pathlib import Path

# Save location
SAVE_DIR = Path(
    "Tensorflow/workspace/images/collectedimages/yes"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Start numbering at 21
image_number = 21
target_images = 20

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nYES gesture image collection started!")
print("Show THUMBS-UP.")
print("Press S to save an image.")
print("Press Q to quit.\n")

while image_number <= 40:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Display instructions
    cv2.putText(
        frame,
        f"YES Images: {image_number - 21}/{target_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Show THUMBS-UP | Press S to save | Q to quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect YES Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        filename = f"yes_{image_number:03d}.jpg"
        filepath = SAVE_DIR / filename

        # Save the mirrored frame
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
