import cv2 # Not strictly needed for PIL-based transform, but good to keep if used elsewhere
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
# torchvision.transforms is not strictly needed for your custom transform, but good general import
from torchvision import transforms
import os

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x.clone().detach()
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.cat((x, res), dim=1)
        return x
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BottleNeckBlock(in_channels + i * growth_rate, growth_rate)) # core idea of dense block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(2 * growth_rate)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense_blocks = nn.ModuleList()
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(num_layers, in_channels, growth_rate))
            in_channels += num_layers * growth_rate # this is the output channels of the dense block and stack channel of the block above
            if i != len(num_blocks) - 1: # Check if not the last block
                # Transition layer
                # Reduce the number of channels by half and downsample
                out_channels = in_channels // 2
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ))
                in_channels = out_channels
        self.batch_norm2 = nn.BatchNorm2d(in_channels)
        self.avgpool2 = nn.AvgPool2d(kernel_size=7)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpooling1(x)

        for block in self.dense_blocks:
            x = block(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Your Custom Transformation Function ---
def transform_img(img_pil, size=(224, 224)): # Takes PIL image as input
    img_pil = img_pil.resize(size)
    img_np = np.array(img_pil)[..., :3]  # Ensure 3 channels
    img_np = img_np / 255.0
    # Convert to CxHxW for PyTorch
    normalized_img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
    return normalized_img_tensor

# --- Configuration ---
NUM_BLOCKS = [6, 12, 24, 16]  # Example for DenseNet-121 like structure - ADJUST TO YOUR MODEL
GROWTH_RATE = 32              # Example growth rate - ADJUST TO YOUR MODEL
NUM_CLASSES = 6               # Must match your model's output and MAP_DICT - ADJUST TO YOUR MODEL

PYTORCH_MODEL_PATH = "./model/best_model_scene_classification.pth" # !!! REPLACE WITH YOUR MODEL PATH !!!
IMAGE_SIZE = (224, 224) # Used by your transform_img function

ROOT_DIR = "./img_cls_scenes_classification/scenes_classification"
train_dir = os.path.join(ROOT_DIR, "train")
test_dir = os.path.join(ROOT_DIR, "val")

MAP_DICT = {
    label_idx : label for label_idx, label in enumerate(os.listdir(train_dir))
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading (Cached for efficiency - UNCHANGED) ---
@st.cache_resource
def load_pytorch_model(model_path, num_blocks_config, growth_rate_config, num_classes_config):
    try:
        model = DenseNet(num_blocks=num_blocks_config, growth_rate=growth_rate_config, num_classes=num_classes_config)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model weights file not found at {model_path}. Please update PYTORCH_MODEL_PATH.")
        return None
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

model = load_pytorch_model(PYTORCH_MODEL_PATH, NUM_BLOCKS, GROWTH_RATE, NUM_CLASSES)

# --- MODIFIED Image Preprocessing Function for Streamlit ---
def preprocess_image_for_streamlit(image_pil): # Takes PIL image
    """Preprocesses a PIL image using your custom transform_img function."""
    transformed_tensor = transform_img(image_pil, size=IMAGE_SIZE)
    return transformed_tensor.unsqueeze(0) # Add batch dimension

# --- Streamlit App UI ---
st.title("🏞️ DenseNet Image Classification")
st.write("Upload an image and the PyTorch DenseNet model will predict its category.")
st.markdown("---")
st.warning("""
    **Important Note on Normalization:** This app uses your custom image transformation,
    which scales pixel values to the range [0, 1] by dividing by 255.0.
    For accurate predictions, the loaded PyTorch model **must** have been trained
    with this exact same normalization method.
""")
st.markdown("---")


uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", width=300)
        st.markdown("---")

        preprocessed_tensor = preprocess_image_for_streamlit(image_pil)
        preprocessed_tensor = preprocessed_tensor.to(DEVICE)

        generate_pred_button = st.button("🔍 Generate Prediction")

        if generate_pred_button and model is not None:
            with st.spinner('Predicting...'):
                with torch.no_grad():
                    outputs = model(preprocessed_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class_index = torch.argmax(probabilities, dim=1).item()
                
                predicted_label = MAP_DICT.get(predicted_class_index, "Unknown Label")
                confidence = probabilities[0][predicted_class_index].item()

            st.success(f"✨ Prediction: **{predicted_label}** (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Make sure you have provided the correct PYTORCH_MODEL_PATH and that the model parameters (NUM_BLOCKS, GROWTH_RATE, NUM_CLASSES) match your trained model.")

elif model is None:
    st.warning("Model could not be loaded. Please check the configuration and model path.")

st.markdown("---")
st.info(f"This app uses a PyTorch DenseNet model to classify images into {NUM_CLASSES} categories: {', '.join(MAP_DICT.values())}.")