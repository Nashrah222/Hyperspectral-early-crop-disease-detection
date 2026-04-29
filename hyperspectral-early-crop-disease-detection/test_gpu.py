import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

_cuda_paths = [
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
]
os.environ['PATH'] = ';'.join(_cuda_paths) + ';' + os.environ.get('PATH', '')

import tensorflow as tf
print("TF Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)

if gpus:
    try:
        # Simple matrix multiplication to force GPU context
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("Matrix Mutlipication Result:\n", c)
            print("[SUCCESS] GPU is fully accessible.")
    except Exception as e:
        print("[FAILURE] GPU caught error:", e)
else:
    print("[FAILURE] No GPU detected.")
