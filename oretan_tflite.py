import numpy as np

import tensorflow as tf
# 2021-12-31 08:09:52.260598: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0

interpreter = tf.lite.Interpreter(model_path="./mobilenetv2_batch32_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_details
 
# [{'name': 'serving_default_input_2:0',
#   'index': 0,
#   'shape': array([  1, 160, 160,   3], dtype=int32),
#   'shape_signature': array([ -1, 160, 160,   3], dtype=int32),
#   'dtype': numpy.float32,
#   'quantization': (0.0, 0),
#   'quantization_parameters': {'scales': array([], dtype=float32),
#    'zero_points': array([], dtype=int32),
#    'quantized_dimension': 0},
#   'sparsity_parameters': {}}]

output_details
 
# [{'name': 'StatefulPartitionedCall:0',
#   'index': 180,
#   'shape': array([1, 1], dtype=int32),
#   'shape_signature': array([-1,  1], dtype=int32),
#   'dtype': numpy.float32,
#   'quantization': (0.0, 0),
#   'quantization_parameters': {'scales': array([], dtype=float32),
#    'zero_points': array([], dtype=int32),
#    'quantized_dimension': 0},
#   'sparsity_parameters': {}}]

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
# [[2.3108282]]

output_data
# array([[2.3108282]], dtype=float32)

tf.constant([1.0], shape=(1,10), dtype=tf.float32)
# 2021-12-31 08:16:25.042744: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
# 2021-12-31 08:16:25.091029: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
# 2021-12-31 08:16:25.140270: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2021-12-31 08:16:25.141372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
# pciBusID: 0000:01:00.0 name: GeForce GTX 1050 computeCapability: 6.1
# coreClock: 1.493GHz coreCount: 5 deviceMemorySize: 3.95GiB deviceMemoryBandwidth: 104.43GiB/s
# 2021-12-31 08:16:25.141405: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
# 2021-12-31 08:16:25.213510: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
# 2021-12-31 08:16:25.213577: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
# 2021-12-31 08:16:25.222110: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
# 2021-12-31 08:16:25.245097: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
# 2021-12-31 08:16:25.245738: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
# 2021-12-31 08:16:25.256934: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
# 2021-12-31 08:16:25.257477: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
# 2021-12-31 08:16:25.257533: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1757] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
# 2021-12-31 08:16:25.258483: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2021-12-31 08:16:25.259506: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
# 2021-12-31 08:16:25.259581: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2021-12-31 08:16:25.259613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      

# <tf.Tensor: shape=(1, 10), dtype=float32, numpy=array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=float32)>


interpreter

# <tensorflow.lite.python.interpreter.Interpreter at 0x7efcb58dae10>


# input_data

# array([[[[0.9791112 , 0.88542897, 0.87605107],
#          [0.40972757, 0.2577797 , 0.7910278 ],
#          [0.00512015, 0.9511606 , 0.84734416],
#          ...,
#          [0.35880226, 0.577569  , 0.48299423],
#          [0.9270226 , 0.8801804 , 0.00280537],
#          [0.3227235 , 0.6516285 , 0.7098169 ]],

#         [[0.9587075 , 0.48431048, 0.0096191 ],
#          [0.76940954, 0.7857148 , 0.9331321 ],
#          [0.4300866 , 0.15123722, 0.9151381 ],
#          ...,
#          [0.82483244, 0.21278277, 0.7978462 ],
#          [0.1492893 , 0.90684885, 0.97229415],
#          [0.0222822 , 0.27458194, 0.47089246]],

#         [[0.9600378 , 0.8670026 , 0.36675578],
#          [0.02592152, 0.6384621 , 0.7215937 ],
#          [0.13668866, 0.58010924, 0.5885671 ],
#          ...,
#          [0.9018993 , 0.94045866, 0.8559659 ],
#          [0.3523581 , 0.5767169 , 0.5497117 ],
#          [0.35278141, 0.7029663 , 0.6890996 ]],

#         ...,

#         [[0.6465092 , 0.1772125 , 0.6615298 ],
#          [0.0086606 , 0.36181462, 0.981482  ],
#          [0.095869  , 0.38347754, 0.39250892],
#          ...,
#          [0.81424034, 0.80370224, 0.04748232],
#          [0.793273  , 0.80448437, 0.5876044 ],
#          [0.9644972 , 0.975504  , 0.40102828]],

#         [[0.49104646, 0.21405838, 0.7022548 ],
#          [0.07115459, 0.97357297, 0.20435278],
#          [0.11858389, 0.6973981 , 0.87645715],
#          ...,
#          [0.04341816, 0.46422043, 0.4306444 ],
#          [0.48127788, 0.6587471 , 0.13058558],
#          [0.14951287, 0.27108294, 0.1093972 ]],

#         [[0.3683209 , 0.5383026 , 0.40826336],
#          [0.23057488, 0.03342762, 0.74881953],
#          [0.9867032 , 0.99964714, 0.45214447],
#          ...,
#          [0.66875523, 0.78947055, 0.69934285],
#          [0.5644578 , 0.27497846, 0.8144627 ],
#          [0.34821552, 0.7934282 , 0.43563285]]]], dtype=float32)

# input_data.shape

# (1, 160, 160, 3)
