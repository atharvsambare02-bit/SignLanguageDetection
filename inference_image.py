import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

MODEL_PATH = "Tensorflow/workspace/exported_model/saved_model"
IMAGE_DIR = Path("Tensorflow/workspace/images/test")
LABEL_MAP = {
    1: "hello",
    2: "yes",
    3: "no",
    4: "thank_you",
    5: "i_love_you"
}

THRESHOLD = 0.30

# Load model
print("Loading trained model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Find test image
images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.jpeg"))
if not images:
    raise FileNotFoundError("No test images found.")

image_path = images[0]
print("Testing image:", image_path)

# Load image
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

input_tensor = tf.convert_to_tensor(image_np[None, ...], dtype=tf.uint8)

# Run inference
results = infer(input_tensor=input_tensor)

boxes = results["detection_boxes"][0].numpy()
classes = results["detection_classes"][0].numpy()
scores = results["detection_scores"][0].numpy()
num = int(results["num_detections"][0].numpy())

# Draw predictions
draw = ImageDraw.Draw(image)
width, height = image.size

print("\nDetection results:")

for i in range(num):
    score = float(scores[i])

    if score < THRESHOLD:
        continue

    class_id = int(classes[i])
    label = LABEL_MAP.get(class_id, f"class_{class_id}")

    ymin, xmin, ymax, xmax = boxes[i]

    x1 = int(xmin * width)
    y1 = int(ymin * height)
    x2 = int(xmax * width)
    y2 = int(ymax * height)

    text = f"{label}: {score:.2%}"

    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    draw.text((x1, max(0, y1 - 20)), text, fill="red")

    print(text)

output_path = "prediction_output.jpg"
image.save(output_path)

print("\nSaved prediction image:", output_path)
