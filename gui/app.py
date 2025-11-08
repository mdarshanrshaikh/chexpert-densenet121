import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# Configuration
MODEL_WEIGHTS_PATH = './densenet121_chexpert_tl_weights.pth' 
NUM_CLASSES = 5
IMAGE_SIZE = 224
TARGET_LABELS = [
    'Atelectasis', 
    'Cardiomegaly', 
    'Edema', 
    'Consolidation', 
    'Pleural Effusion'
]
DEVICE = torch.device("cpu") 

# Model Definition and Loading

# Function to setup the DenseNet model architecture (must match training)
@st.cache_resource
def load_model(weights_path):
    # Load pre-trained DenseNet121 model
    model = models.densenet121(weights=None) # We don't load ImageNet weights here

    # Replace the final classification layer
    num_ftrs = model.classifier.in_features
    
    # New classifier head
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES),
        nn.Sigmoid() 
    )
    
    # Load the trained state dictionary
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.success("Model weights loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model weights file not found at {weights_path}. Please train and save the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
        
    return model

# Image Preprocessing

# Transforms
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Prediction Function

def predict_image(model, image):
    transform = get_image_transforms()
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        
    # Convert output probabilities to NumPy array
    probabilities = output.cpu().squeeze().numpy()
    
    return probabilities

# STREAMLIT APP LAYOUT

st.title("👨‍⚕️ CheXpert DenseNet121 Prediction Demo")
st.markdown("Upload a chest X-ray image to get predictions for the 5 benchmark pathologies using a PyTorch Transfer Learning model.")

# Load the model once
model = load_model(MODEL_WEIGHTS_PATH)

if model:
    uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded X-ray', use_column_width=True)
        st.write("")
        st.subheader("Model Prediction Results")
        
        # Run prediction
        with st.spinner('Analyzing X-ray with DenseNet121...'):
            probabilities = predict_image(model, image)
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame({
            'Pathology': TARGET_LABELS,
            'Probability': probabilities
        })
        
        # Sort and format the results
        results_df = results_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
        results_df['Probability'] = (results_df['Probability'] * 100).round(2).astype(str) + '%'
        
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        
        # Visualization (Bar Chart)
        st.subheader("Probability Visualization")
        
        # Convert probability column back to float for charting
        chart_data = results_df.copy()
        chart_data['Probability_Val'] = chart_data['Probability'].str.replace('%', '').astype(float) / 100
        
        st.bar_chart(chart_data.set_index('Pathology')['Probability_Val'])