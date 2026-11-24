import runpy
import sys
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# Compatibility fixes for TF-Slim 1.1.0 with TensorFlow 2.15.1
control_flow_ops.case = tf.case
control_flow_ops.cond = tf.cond

print("Evaluation compatibility patches applied: case and cond")

sys.argv[0] = "Tensorflow/models/research/object_detection/model_main_tf2.py"

runpy.run_path(
    "Tensorflow/models/research/object_detection/model_main_tf2.py",
    run_name="__main__"
)
