import torch.quantization

# Load your trained FP32 model
model_fp32 = BERT4Rec(...)
model_fp32.load_state_dict(torch.load("model.pth"))

# Quantize weights to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {nn.Linear, nn.Embedding}, # Specify layers to quantize
    dtype=torch.qint8
)