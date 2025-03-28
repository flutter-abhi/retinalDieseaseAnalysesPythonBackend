import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from src.models.model import OcularDiseaseModel
import torchvision.models as models
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
import plotly.graph_objects as go

# Add numpy scalar to safe globals
add_safe_globals([scalar])

# Define disease classes
DISEASE_CLASSES = [
    'Normal (N)',
    'Diabetes (D)',
    'Glaucoma (G)',
    'Cataract (C)',
    'AMD (A)',
    'Hypertension (H)',
    'Myopia (M)',
    'Other (O)'
]

def load_model(model_path='best_model.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OcularDiseaseModel().to(device)
    
    try:
        # First try loading with weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        st.warning(f"Could not load with weights_only=True: {str(e)}")
        st.info("Attempting to load without weights_only restriction...")
        # If that fails, load without weights_only restriction
        checkpoint = torch.load(model_path, map_location=device)
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess the uploaded image"""
    # Define the transform
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transform
    transformed = transform(image=np.array(image))
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def predict(model, left_image, right_image, device):
    """Make predictions using the model"""
    with torch.no_grad():
        # Preprocess images
        left_tensor = preprocess_image(left_image).to(device)
        right_tensor = preprocess_image(right_image).to(device)
        
        # Get predictions
        outputs = model(left_tensor, right_tensor)
        
        # Properly convert logits to probabilities with sigmoid
        predictions = {}
        for key, logits in outputs.items():
            # Apply sigmoid and convert to numpy
            probs = torch.sigmoid(logits)
            # Ensure values are between 0 and 1
            probs = torch.clamp(probs, 0, 1)
            # Convert to numpy array
            predictions[key] = probs.cpu().numpy()[0]
        
        return predictions

def display_predictions(predictions, title):
    """Display predictions in a bar chart"""
    # Ensure predictions are valid probabilities
    valid_predictions = np.clip(predictions, 0, 1)
    
    fig = go.Figure(data=[
        go.Bar(x=DISEASE_CLASSES, y=valid_predictions)
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Disease",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed results with proper percentage formatting
    st.write("Detailed Results:")
    for disease, prob in zip(DISEASE_CLASSES, valid_predictions):
        # Format as percentage between 0% and 100%
        st.write(f"{disease}: {prob*100:.2f}%")

def main():
    st.set_page_config(
        page_title="Ocular Disease Detection",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Ocular Disease Detection")
    st.write("""
    This application uses a deep learning model to detect various ocular diseases from retinal images.
    Please upload both left and right eye images for accurate diagnosis.
    """)
    
    # Add instructions
    with st.expander("Instructions"):
        st.write("""
        1. Upload a clear image of the left eye
        2. Upload a clear image of the right eye
        3. Click the 'Predict Diseases' button
        4. View the results for each eye separately and the combined diagnosis
        """)
    
    # Load the model
    try:
        model, device = load_model()
        st.success(f"Model loaded successfully! Using device: {device}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Left Eye")
        left_image = st.file_uploader("Upload left eye image", type=['jpg', 'jpeg', 'png'])
        if left_image:
            left_image = Image.open(left_image)
            st.image(left_image, caption="Left Eye", use_column_width=True)
    
    with col2:
        st.subheader("Right Eye")
        right_image = st.file_uploader("Upload right eye image", type=['jpg', 'jpeg', 'png'])
        if right_image:
            right_image = Image.open(right_image)
            st.image(right_image, caption="Right Eye", use_column_width=True)
    
    # Add a predict button
    if st.button("Predict Diseases", type="primary"):
        if left_image and right_image:
            try:
                with st.spinner("Processing images and making predictions..."):
                    # Get predictions
                    predictions = predict(model, left_image, right_image, device)
                    
                    # Display results for each eye
                    st.subheader("Left Eye Diagnosis")
                    display_predictions(predictions['left'], "Left Eye Disease Probabilities")
                    
                    st.subheader("Right Eye Diagnosis")
                    display_predictions(predictions['right'], "Right Eye Disease Probabilities")
                    
                    st.subheader("Combined Diagnosis")
                    display_predictions(predictions['combined'], "Combined Disease Probabilities")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please upload both left and right eye images")

if __name__ == "__main__":
    main() 


#     Flask==3.0.0
# Flask-Cors==5.0.0
# numpy==1.26.1
# pandas==2.1.1
# scikit-learn==1.4.2
# requests==2.31.0
# matplotlib==3.8.4
# pymongo==4.6.1
# python-dotenv==1.0.1