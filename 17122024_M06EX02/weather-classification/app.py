import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

# Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), # stride is the same maxpooling and change the same channels
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    # n_block_list is a list of integers. Each integer specifies how many residual blocks to use in each of the four main stages of the ResNet. 
    def __init__(self, residual_block, n_block_list, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.create_layer(residual_block, 64, 64, n_block_list[0], stride=1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_block_list[1], stride=2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_block_list[2], stride=2)
        self.conv5 = self.create_layer(residual_block, 256, 512, n_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, num_classes)
    
    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)
        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride=1)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)
        return block_sequential
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

# --- Streamlit App Configuration ---
MODEL_PATH = './model/best_model_weather_classification.pth' # CHANGE THIS TO YOUR .pth FILE NAME

# Directory containing the dataset
ROOT_DIR = "./img_cls_weather_dataset/weather-dataset/dataset"

# Creating classes dictionary
CLASSES = {
    label_idx : class_name for label_idx, class_name in enumerate(sorted(os.listdir(ROOT_DIR)))
}
CLASS_NAMES = list(CLASSES.values())
NUM_CLASSES = len(CLASS_NAMES) # Should be 11 based on the list above

# Define the ResNet architecture (e.g., for ResNet18-like structure)
# YOU MUST ADJUST THIS TO MATCH YOUR TRAINED MODEL'S CONFIGURATION
N_BLOCK_LIST = [2, 2, 2, 2] # Example: ResNet18 [2 blocks in conv2, 2 in conv3, etc.]

# Image preprocessing transformations
# Adjust mean and std if your model was trained with different normalization
IMG_SIZE = (224, 224)
PREPROCESS_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), # Converts PIL image to PyTorch tensor (HWC to CHW, scales to [0,1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
])

# Descriptions for each weather phenomenon.
WEATHER_DESCRIPTIONS = {
    'hail': "Precipitation in the form of small balls or lumps of ice, typically causing significant damage to crops and property.",
    'rainbow': "An optical phenomenon occurring when sunlight interacts with water droplets, creating a spectrum of colors in the sky.",
    'frost': "A deposit of small white ice crystals formed when surfaces are cooled below the dew point.",
    'rime': "A white ice deposit formed when supercooled water droplets freeze upon impact with cold surfaces.",
    'fogsmog': "Reduced visibility due to either natural fog or pollution-induced smog in the atmosphere.",
    'snow': "Precipitation in the form of small white ice crystals, forming a white layer when accumulated.",
    'rain': "Liquid precipitation falling from clouds in the form of water drops.",
    'glaze': "A smooth, transparent ice coating occurring when supercooled rain freezes on contact with surfaces.",
    'lightning': "A sudden electrostatic discharge during a thunderstorm, creating visible plasma.",
    'sandstorm': "A meteorological phenomenon where strong winds lift and transport sand particles, reducing visibility.",
    'dew': "Water droplets formed by condensation of water vapor on cool surfaces, typically occurring in the morning."
}

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_pytorch_model():
    """Load the trained PyTorch ResNet model"""
    try:
        # Instantiate the model with the correct architecture
        model = ResNet(ResidualBlock, N_BLOCK_LIST, NUM_CLASSES)
        # Load the saved state dictionary
        # Ensure map_location is used for flexibility if model was saved on GPU and run on CPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE) # Move model to the determined device
        model.eval() # Set the model to evaluation mode
        st.success(f"PyTorch ResNet model loaded successfully from {MODEL_PATH} on {DEVICE}.")
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure the path is correct.")
    except Exception as e:
        raise RuntimeError(f"Could not load PyTorch model from {MODEL_PATH}: {str(e)}")

def preprocess_pil_image(image_pil):
    """
    Preprocess the uploaded PIL image for PyTorch model prediction.
    Args:
        image_pil (PIL.Image.Image): The image uploaded by the user.
    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor, ready for the model.
    """
    # Convert RGBA or Grayscale to RGB if needed
    if image_pil.mode == 'RGBA' or image_pil.mode == 'LA' or (image_pil.mode == 'P' and 'transparency' in image_pil.info):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode == 'L': # Grayscale
         image_pil = image_pil.convert('RGB')


    # Apply the defined transformations
    tensor_image = PREPROCESS_TRANSFORMS(image_pil)
    
    # Add batch dimension (BCHW)
    batch_tensor_image = tensor_image.unsqueeze(0)
    return batch_tensor_image

def predict_weather_pytorch(model, processed_image_tensor):
    """
    Make a prediction on the preprocessed PyTorch tensor.
    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        processed_image_tensor (torch.Tensor): The image after preprocessing.
    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    processed_image_tensor = processed_image_tensor.to(DEVICE) # Ensure tensor is on the correct device
    
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(processed_image_tensor)
        
        # Apply Softmax to get probabilities if model outputs logits
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get the confidence score (highest probability) and predicted class index
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class_index = predicted_idx.item()
    confidence_score = confidence.item()
    
    if 0 <= predicted_class_index < len(CLASS_NAMES):
        predicted_class_name = CLASS_NAMES[predicted_class_index]
    else:
        predicted_class_name = "Unknown"
        st.warning(f"Model predicted an out-of-bounds class index: {predicted_class_index}")

    return predicted_class_name, confidence_score

def main():
    st.set_page_config(
        page_title="Weather Phenomenon Classifier (PyTorch)",
        page_icon="🌦️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌈 Weather Phenomenon Classifier (PyTorch ResNet)")
    st.markdown("Upload an image to identify the weather phenomenon it depicts!")
    
    # Load model
    try:
        model = load_pytorch_model()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()
        
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        image_pil_display = None # To store the PIL image for display
        if uploaded_file is not None:
            try:
                image_pil_display = Image.open(uploaded_file)
                st.image(image_pil_display, caption='Uploaded Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error opening or displaying image: {str(e)}")
                uploaded_file = None 

    with col2:
        st.subheader("Analysis Results")
        # Check if image_pil_display is not None before trying to use it
        if uploaded_file is not None and image_pil_display is not None:
            if st.button('🔍 Analyze Weather Phenomenon', key="analyze_button"):
                with st.spinner('🧠 Analyzing image... Please wait.'):
                    try:
                        # Preprocess image
                        processed_image_tensor = preprocess_pil_image(image_pil_display)
                        
                        # Get prediction
                        weather_class, confidence = predict_weather_pytorch(model, processed_image_tensor)
                        
                        st.success(f"✅ Analysis complete!")
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric(label="Identified Phenomenon", value=weather_class.title())
                        with res_col2:
                            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                        
                        description = WEATHER_DESCRIPTIONS.get(weather_class, "No description available.")
                        st.info(f"**About {weather_class.title()}:** {description}")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
        elif uploaded_file is None and st.session_state.get("analyze_button"): # Check if button was pressed without file
             st.warning("Please upload an image first.")
        else:
            st.info("Upload an image and click 'Analyze Weather Phenomenon' to see the results here.")

    with st.expander("ℹ️ About This App", expanded=False):
        st.write("""
        This application uses a PyTorch ResNet deep learning model
        trained on a diverse dataset of weather images to classify various weather phenomena.
        
        **Identifiable Weather Conditions:**
        """)
        for phenomenon, desc_snippet in {
            '🌨️ Hail': "Frozen precipitation", '🌈 Rainbow': "Optical phenomenon",
            '❄️ Frost': "Surface ice crystal", '🌫️ Rime': "Deposited ice",
            '😶‍🌫️ Fog/Smog': "Reduced visibility", '🌨️ Snow': "Crystalline precipitation",
            '🌧️ Rain': "Liquid precipitation", '🧊 Glaze': "Surface ice coating",
            '⚡ Lightning': "Electrical discharge", '🌪️ Sandstorm': "Wind-transported sand",
            '💧 Dew': "Surface water condensation"
        }.items():
            st.markdown(f"* **{phenomenon}**: {desc_snippet}")
        st.write("Upload an image and click 'Analyze' to get a prediction.")
        
    st.sidebar.title("📸 Image Upload Guidelines")
    st.sidebar.info("""
    - **Clarity:** Upload clear, well-lit images.
    - **Visibility:** Ensure the phenomenon is clearly visible.
    - **Originality:** Avoid heavily edited or filtered images.
    - **Format:** JPG, JPEG, or PNG.
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by PyTorch & Streamlit")

if __name__ == "__main__":
    main()
