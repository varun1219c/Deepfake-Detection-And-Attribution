import torch
import timm
import onnxruntime as ort
import numpy as np
import os
from datetime import datetime

DATA_DIR = r"ImagesDF" 
OUTPUT_DIR = "training_reports"
MODELS_DIR = "saved_models"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_TIMESTAMP)

NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
IMAGE_SIZE = 299  
NUM_WORKERS = 4

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7
LR_REDUCTION_FACTOR = 0.5
GRADIENT_CLIP_VALUE = 1.0

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

USE_PRETRAINED = True
DROPOUT_RATE = 0.5
USE_ATTENTION = True

class XceptionNetWithAttention(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(XceptionNetWithAttention, self).__init__()
        self.base_model = timm.create_model('xception', pretrained=USE_PRETRAINED, num_classes=0)
        in_features = self.base_model.num_features

        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_features // 8, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout_rate * 0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)

        attention_weights = self.attention(features.unsqueeze(-1).unsqueeze(-1))
        features = features * attention_weights.squeeze(-1).squeeze(-1)
        output = self.classifier(features)
        return output

def convert_to_onnx(model_path, onnx_path, num_classes, image_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XceptionNetWithAttention(num_classes=num_classes, dropout_rate=DROPOUT_RATE).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_state = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items() if k in model.state_dict()}
    model.load_state_dict(model_state, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model successfully exported to: {onnx_path}")

        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-3, atol=1e-5)
        print("ONNX model validation passed: outputs match PyTorch.")

    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == '__main__':

    MODEL_PATH = "saved_models/best_model_20251007_160007.pt"
    ONNX_PATH = os.path.join(MODELS_DIR, f"dfda_image_model.onnx")
    NUM_CLASSES = 10  

    convert_to_onnx(MODEL_PATH, ONNX_PATH, NUM_CLASSES, IMAGE_SIZE)