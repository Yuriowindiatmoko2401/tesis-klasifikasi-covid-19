import tensorflow as tf

saved_model_dir = "./mobilenetv2_batch32.tf"
converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

fo = open(
    "./mobilenetv2_batch32_model.tflite", "wb")
fo.write(tflite_model)
fo.close