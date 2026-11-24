
import cv2
import tensorflow as tf
import numpy as np

MODEL_PATH = (
    "Tensorflow/workspace/exported-models/"
    "my_ssd_mobilenet_v2_fpnlite_step3000/saved_model"
)

CLASS_NAMES = {
    1: "hello",
    2: "yes",
    3: "no",
    4: "thank_you",
    5: "i_love_you",
}

CONFIDENCE_THRESHOLD = 0.15

print("Loading trained model...")
model = tf.saved_model.load(MODEL_PATH)
detect_fn = model.signatures["serving_default"]
print("Model loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started!")
print("Show your gestures to the camera.")
print("Press Q to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(
        rgb_frame[np.newaxis, ...],
        dtype=tf.uint8
    )

    detections = detect_fn(input_tensor=input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)
    scores = detections["detection_scores"][0].numpy()
    num_detections = int(detections["num_detections"][0].numpy())

    frame_count += 1

    if frame_count % 30 == 0:
        print("Top scores:", scores[:5])
        print("Predicted classes:", classes[:5])

    height, width, _ = frame.shape

    best_index = -1
    best_score = 0.0

    for i in range(num_detections):
        if scores[i] < CONFIDENCE_THRESHOLD:
            continue

        if classes[i] not in CLASS_NAMES:
            continue

        if scores[i] > best_score:
            best_score = scores[i]
            best_index = i

    if best_index != -1:
        ymin, xmin, ymax, xmax = boxes[best_index]

        x1 = max(0, int(xmin * width))
        y1 = max(0, int(ymin * height))
        x2 = min(width - 1, int(xmax * width))
        y2 = min(height - 1, int(ymax * height))

        label = CLASS_NAMES[classes[best_index]]
        confidence = best_score * 100

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 3
        )

        cv2.putText(
            frame,
            f"{label}: {confidence:.1f}%",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam detection stopped.")
