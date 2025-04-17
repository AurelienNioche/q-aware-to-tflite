import keras
import nobuco
import tensorflow as tf
import torch
from nobuco import ChannelOrder, ChannelOrderingStrategy
# from nobuco.converters.type_cast import dtype_pytorch2keras # Not needed if not mapping types
# from tensorflow.lite.python.lite import TFLiteConverter # Use tf.lite directly
from torch import nn # , ops # ops not needed if not using quantized ops
import numpy as np

# # Set the quantized engine for Apple Silicon - Not needed if not using PyTorch quantized ops
# if torch.backends.mps.is_available():
#     try:
#         torch.backends.quantized.engine = 'qnnpack'
#         print("Quantized engine set to 'qnnpack'")
#     except AttributeError:
#         print("Warning: Could not set quantized engine. This PyTorch version might not support it.")
# else:
#     # You might want a different default for non-MPS systems if needed
#     pass


class Identity(nn.Module):
    def forward(self, x):
        return x

# Standard Float Model - Remove PyTorch Quantization
class Model(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        # Store standard float weights and bias
        self.register_parameter('weight', nn.Parameter(weight))
        if bias is not None:
            self.register_parameter('bias', nn.Parameter(bias))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        # Use standard nn.Linear
        x = nn.functional.linear(x, self.weight, self.bias)
        return x

# --- Remove Nobuco converters for quantized ops --- 
# @nobuco.converter(...)
# def converter_quantize_per_tensor(...):
#     ...

# @nobuco.converter(...)
# def converter_dequantize(...):
#     ...

# @nobuco.converter(...)
# def converter_linear_quantized(...):
#     ...
# --- Nobuco will use its default converter for nn.Linear --- 

weight_float = torch.rand((100, 100))
bias_float = torch.rand((100,))
model = Model(weight_float, bias_float)

x_float = torch.rand(size=(1, 100)) * 200 - 100

# Convert the FLOAT model using Nobuco
print("Converting float PyTorch model to Keras...")
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x_float],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)
print("Keras float model conversion successful.")

model_path_float = "float_model"
keras_model.save(model_path_float + ".h5")
print(f"Keras float model saved to {model_path_float}.h5")

# --- TFLite Post-Training Quantization --- 

print("\nStarting TFLite conversion with Post-Training Quantization...")

# Define the representative dataset using FLOAT data
def representative_dataset_gen():
  for _ in range(100): # Use a reasonable number of samples
    # Generate float data representative of the actual input range
    yield [ (torch.rand(size=(1, 100)) * 200 - 100).numpy().astype(np.float32) ]

# Convert the Keras FLOAT model to TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Enable default optimizations (includes quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide the representative dataset (essential for calibration)
converter.representative_dataset = representative_dataset_gen

# Ensure integer-only quantization is enforced
# This requires representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Specify input/output types for the final TFLite model
converter.inference_input_type = tf.int8  # Model will expect int8 input
converter.inference_output_type = tf.int8 # Model will produce int8 output


try:
    print("Running TFLite converter...")
    tflite_quant_model = converter.convert()
    output_quant_path = "model_int8_quant.tflite"
    with open(output_quant_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Successfully converted and saved INT8 TFLite model to {output_quant_path}")

    # --- Verification --- 
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_quant_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite Input Details:", input_details)
    print("TFLite Output Details:", output_details)

    # Prepare sample input: quantize float input according to TFLite input specs
    sample_float_input = (torch.rand(size=(1, 100)) * 200 - 100).numpy().astype(np.float32)
    input_scale, input_zero_point = input_details['quantization']
    print(f"Input scale: {input_scale}, Input zero-point: {input_zero_point}")
    if input_scale == 0 and input_zero_point == 0:
        print("Warning: Input quantization parameters are zero. Input might not be quantized.")
        tflite_input = sample_float_input.astype(input_details['dtype'])
    else:
        tflite_input = (sample_float_input / input_scale + input_zero_point).astype(input_details['dtype'])


    # Run inference
    interpreter.set_tensor(input_details['index'], tflite_input)
    interpreter.invoke()
    tflite_output_quant = interpreter.get_tensor(output_details['index'])

    # Dequantize output for comparison
    output_scale, output_zero_point = output_details['quantization']
    print(f"Output scale: {output_scale}, Output zero-point: {output_zero_point}")
    if output_scale == 0 and output_zero_point == 0:
        print("Warning: Output quantization parameters are zero. Output might not be quantized.")
        tflite_output_float = tflite_output_quant.astype(np.float32)
    else:
        tflite_output_float = (tflite_output_quant.astype(np.float32) - output_zero_point) * output_scale

    print("Sample TFLite Input (quantized):")
    print(tflite_input)
    print("\nSample TFLite Output (dequantized):")
    print(tflite_output_float)

    # Compare with original PyTorch float model output (optional)
    # with torch.no_grad():
    #    pytorch_output = model(torch.from_numpy(sample_float_input)).numpy()
    # print("\nOriginal PyTorch output (float):")
    # print(pytorch_output)
    # diff = np.mean(np.abs(pytorch_output - tflite_output_float))
    # print(f"\nMean Absolute Difference: {diff}")


except Exception as e:
    print(f"\nError during TFLite conversion or verification: {e}")
    import traceback
    traceback.print_exc()
