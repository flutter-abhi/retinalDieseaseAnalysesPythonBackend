from flask import Blueprint, request, jsonify
import torch
from PIL import Image
import io
import requests
from app.models.model import get_model
from app.utils.preprocessing import preprocess_image
import urllib.request

api = Blueprint('api', __name__)

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

def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        raise Exception(f"Error loading image from URL: {str(e)}")

@api.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    # Check if URLs are provided
    if 'left_eye_url' not in data or 'right_eye_url' not in data:
        return jsonify({'error': 'Both left and right eye image URLs are required'}), 400
    
    try:
        # Load images from URLs
        left_image = load_image_from_url(data['left_eye_url'])
        right_image = load_image_from_url(data['right_eye_url'])
        
        # Get model and device
        model, device = get_model()
        
        # Preprocess images
        left_tensor = preprocess_image(left_image).to(device)
        right_tensor = preprocess_image(right_image).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(left_tensor, right_tensor)
            
            # Convert logits to probabilities
            predictions = {}
            for key, logits in outputs.items():
                probs = torch.sigmoid(logits)
                probs = torch.clamp(probs, 0, 1)
                predictions[key] = probs.cpu().numpy()[0].tolist()
        
        # Format response
        response = {
            'predictions': {
                'left_eye': dict(zip(DISEASE_CLASSES, predictions['left'])),
                'right_eye': dict(zip(DISEASE_CLASSES, predictions['right'])),
                'combined': dict(zip(DISEASE_CLASSES, predictions['combined']))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500