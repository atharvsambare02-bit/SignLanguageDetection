
import cv2
import os
import time

IMAGE_PATH = "Tensorflow/workspace/images/collectedimages"

labels = ["hello", "yes", "no", "thank_you", "i_love_you"]

TARGET_IMAGES = 20

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

for label in labels:

    folder = os.path.join(IMAGE_PATH, label)
    os.makedirs(folder, exist_ok=True)

    existing = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ])

    print(f"\nCollecting images for: {label}")
    print("Show the gesture and press C to capture.")
    print("Press Q to quit.")

    time.sleep(2)

    while existing < TARGET_IMAGES:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame not received!")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"{label}: {existing}/{TARGET_IMAGES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign Language Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            filename = f"{label}_{existing + 1:03d}.jpg"
            filepath = os.path.join(folder, filename)

            cv2.imwrite(filepath, frame)
            existing += 1

            print(f"Saved {filename}")

        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"Completed: {label}")

cap.release()
cv2.destroyAllWindows()

print("\nImage collection completed!")
