
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = (
    "Tensorflow/workspace/exported-models/"
    "my_ssd_mobilenet_v2_fpnlite/saved_model"
)

IMAGE_PATH = "test_image.jpg"
THRESHOLD = 0.10
CATEGORY_INDEX = {
    1: "hello",
    2: "yes",
    3: "no",
    4: "thank_you",
    5: "i_love_you",
}

print("Loading exported model...")

detect_fn = tf.saved_model.load(MODEL_PATH)

print("Model loaded successfully!")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_tensor = tf.convert_to_tensor(image_rgb)
input_tensor = input_tensor[tf.newaxis, ...]

print("Running detection...")

detections = detect_fn(input_tensor)

num_detections = int(detections.pop("num_detections"))

detections = {
    key: value[0, :num_detections].numpy()
    for key, value in detections.items()
}

boxes = detections["detection_boxes"]
scores = detections["detection_scores"]
classes = detections["detection_classes"].astype(np.int32)

height, width, _ = image_rgb.shape
detected_count = 0

print("\n--- RAW MODEL PREDICTIONS ---")

for i in range(min(10, num_detections)):
    class_id = classes[i]
    label = CATEGORY_INDEX.get(class_id, "unknown")

    print(
        f"Prediction {i+1}: "
        f"{label} | "
        f"Confidence: {scores[i]:.4%}"
    )

print("-----------------------------\n")

for i in range(num_detections):
    if scores[i] < THRESHOLD:
        continue

    ymin, xmin, ymax, xmax = boxes[i]

    x1 = int(xmin * width)
    y1 = int(ymin * height)
    x2 = int(xmax * width)
    y2 = int(ymax * height)

    class_id = classes[i]
    label = CATEGORY_INDEX.get(class_id, "unknown")

    confidence = float(scores[i])

    cv2.rectangle(
        image_rgb,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image_rgb,
        f"{label}: {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    print(
        f"Detected: {label} | "
        f"Confidence: {confidence:.2%}"
    )

    detected_count += 1

if detected_count == 0:
    print("No signs detected above the confidence threshold.")

output_image = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite("detection_result.jpg", output_image)

print("Detection result saved as detection_result.jpg")
