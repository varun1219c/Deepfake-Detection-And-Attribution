import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import os
import plotly.graph_objects as go
import plotly.express as px

MODEL_PATH = r"Image.pt"

CLASS_NAMES = ['DALL-E', 'DeepFaceLab', 'Face2Face', 'FaceShifter', 'FaceSwap', 
               'Midjourney', 'NeuralTextures', 'Real', 'Stable Diffusion', 'StyleGAN']

IMAGE_SIZE = 299
DROPOUT_RATE = 0.5
USE_ATTENTION = True

class AttentionBlock(nn.Module):
    """Attention mechanism for XceptionNet."""
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),
            nn.ReLU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_weights = self.attention(x)
        return x * att_weights

class EnhancedXceptionNet(nn.Module):
    """XceptionNet with attention mechanism and improved head."""
    def __init__(self, num_classes, dropout_rate=0.3, use_attention=True):
        super(EnhancedXceptionNet, self).__init__()
        self.base_model = timm.create_model('xception', pretrained=False)
        in_features = self.base_model.get_classifier().in_features
        self.base_model.reset_classifier(0)

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(in_features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        if self.use_attention:
            features = self.attention(features)
        output = self.classifier(features)
        return output

@st.cache_resource
def load_model(model_path, num_classes, device):
    """Loads the PyTorch model checkpoint."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please update MODEL_PATH in the code to point to your .pt file")
        return None

    try:

        model = EnhancedXceptionNet(
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            use_attention=USE_ATTENTION
        )
 
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        st.success(f" Model loaded successfully from: {model_path}")

        if 'val_f1' in checkpoint:
            st.info(f" Model F1-Score: {checkpoint['val_f1']:.4f}")
        if 'val_acc' in checkpoint:
            st.info(f" Model Accuracy: {checkpoint['val_acc']:.4f}")

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocesses the uploaded image to match model's input requirements."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)

def run_inference(model, processed_image, device):
    """Runs inference using the PyTorch model."""
    with torch.no_grad():
        processed_image = processed_image.to(device)
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()

def create_confidence_chart(probabilities, class_names):
    """Creates an interactive bar chart for confidence scores."""
    probs = probabilities[0] * 100  

    sorted_indices = np.argsort(probs)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    colors = ['#2ecc71' if cls == 'Real' else '#e74c3c' for cls in sorted_classes]

    fig = go.Figure(data=[
        go.Bar(
            y=sorted_classes,
            x=sorted_probs,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.2f}%' for p in sorted_probs],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Confidence Scores by Class",
        xaxis_title="Confidence (%)",
        yaxis_title="Class",
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

def create_gauge_chart(confidence, predicted_class):
    """Creates a gauge chart for the main prediction confidence."""

    if predicted_class == 'Real':
        color = '#2ecc71'  
        title_text = "Real Image Confidence"
    else:
        color = '#e74c3c'  
        title_text = f"{predicted_class} Confidence"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title_text, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig

def main():

    st.set_page_config(
        page_title="DeepFake Detector",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: 
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: 
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .real-prediction {
            background-color: 
            border: 2px solid 
            color: 
        }
        .fake-prediction {
            background-color: 
            border: 2px solid 
            color: 
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header"> DeepFake Image Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Image Authenticity Analysis</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header(" About")
        st.write("""
        This application uses an **Enhanced XceptionNet** model with attention mechanisms 
        to detect deepfake images and identify their generation method.
        """)

        st.header(" Model Info")
        st.write(f"**Architecture:** XceptionNet + Attention")
        st.write(f"**Input Size:** {IMAGE_SIZE}x{IMAGE_SIZE}")
        st.write(f"**Classes:** {len(CLASS_NAMES)}")

        st.header(" Supported Classes")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            emoji = "" if class_name == "Real" else ""
            st.write(f"{emoji} {class_name}")

        st.header(" System Info")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"**Device:** {device.type.upper()}")
        if device.type == 'cuda':
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH, len(CLASS_NAMES), device)

    if model is None:
        st.error(" Cannot proceed without a valid model. Please check the model path.")
        st.stop()

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📤 Upload an image to analyze",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")

            st.info(f"**Size:** {image.size[0]} x {image.size[1]} | **Mode:** {image.mode}")

        with col2:
            st.subheader(" Analysis Results")

            with st.spinner("Analyzing image..."):

                processed_image = preprocess_image(image)

                probabilities = run_inference(model, processed_image, device)

                prediction_index = np.argmax(probabilities)
                prediction_confidence = probabilities[0, prediction_index]
                predicted_class = CLASS_NAMES[prediction_index]

            if predicted_class == "Real":
                st.markdown(f"""
                <div class="prediction-box real-prediction">
                    <h2> REAL IMAGE</h2>
                    <h3>Confidence: {prediction_confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box fake-prediction">
                    <h2> DEEPFAKE DETECTED</h2>
                    <h3>Type: {predicted_class}</h3>
                    <h3>Confidence: {prediction_confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(
                create_gauge_chart(prediction_confidence, predicted_class),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader(" Detailed Confidence Analysis")

        fig = create_confidence_chart(probabilities, CLASS_NAMES)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Detailed Probabilities"):
            prob_data = {
                'Class': CLASS_NAMES,
                'Confidence (%)': [f"{p*100:.2f}%" for p in probabilities[0]],
                'Raw Score': [f"{p:.6f}" for p in probabilities[0]]
            }
            st.table(prob_data)

        st.markdown("---")
        if st.button(" Download Analysis Report"):
            report = f"""
DeepFake Detection Analysis Report
{'='*50}

Image: {uploaded_file.name}
Prediction: {predicted_class}
Confidence: {prediction_confidence:.2%}

Detailed Probabilities:
{'-'*50}
"""
            for i, class_name in enumerate(CLASS_NAMES):
                report += f"{class_name}: {probabilities[0, i]*100:.2f}%\n"

            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name=f"analysis_report_{uploaded_file.name}.txt",
                mime="text/plain"
            )

    else:
        st.info(" Please upload an image to begin analysis")

        st.markdown("---")
        st.subheader("💡 Example Use Cases")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Face Manipulation**")
            st.write("Detect face swaps, deepfakes, and face reenactment")

        with col2:
            st.markdown("**AI-Generated Art**")
            st.write("Identify images from DALL-E, Midjourney, Stable Diffusion")

        with col3:
            st.markdown("**Authenticity Verification**")
            st.write("Verify if an image is real or synthetically generated")

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Powered by Enhanced XceptionNet with Attention Mechanism</p>
            <p> Running on PyTorch | Advanced Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()