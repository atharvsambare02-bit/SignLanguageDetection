import runpy
import sys
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# Restore legacy functions expected by tf_slim.
control_flow_ops.case = tf.compat.v1.case
control_flow_ops.cond = tf.compat.v1.cond

print("TensorFlow compatibility patches applied.", flush=True)

script = "Tensorflow/models/research/object_detection/model_main_tf2.py"
sys.argv = [script] + sys.argv[1:]

runpy.run_path(script, run_name="__main__")
