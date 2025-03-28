from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import requests
from utils.preprocessing import preprocess_image
from models.model import OcularDiseaseModel

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = None

DISEASE_CLASSES = [
    'Normal (N)', 'Diabetes (D)', 'Glaucoma (G)', 'Cataract (C)',
    'AMD (A)', 'Hypertension (H)', 'Myopia (M)', 'Other (O)'
]

def load_model():
    """Initialize the PyTorch model"""
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OcularDiseaseModel().to(device)
    
    # Load model weights
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully on {device}")

def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        raise Exception(f"Error loading image from URL: {str(e)}")


@app.route('/', methods=['GET'])
def home():
     return jsonify({'message': 'python backend for retinal diesase analyser started'}), 400

    
    # Get JSON data


@app.route('/predict', methods=['POST'])
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
        
        # Preprocess images
        left_tensor = preprocess_image(left_image).to(device)
        right_tensor = preprocess_image(right_image).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(left_tensor, right_tensor)
            
            # Convert logits to probabilities and then to percentages
            predictions = {}
            for key, logits in outputs.items():
                probs = torch.sigmoid(logits)
                probs = torch.clamp(probs, 0, 1)
                # Convert to percentages
                percentages = (probs.cpu().numpy()[0] * 100).tolist()
                predictions[key] = percentages
        
        # Format response with percentages
        response = {
            'predictions': {
                'left_eye': {
                    disease: f"{percentage:.2f}%" 
                    for disease, percentage in zip(DISEASE_CLASSES, predictions['left'])
                },
                'right_eye': {
                    disease: f"{percentage:.2f}%" 
                    for disease, percentage in zip(DISEASE_CLASSES, predictions['right'])
                },
                'combined': {
                    disease: f"{percentage:.2f}%" 
                    for disease, percentage in zip(DISEASE_CLASSES, predictions['combined'])
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

port_num = process.env.PORT

if __name__ == '__main__':
    # Load the model before starting the server
    load_model()
        # Start the Flask app
    app.run(host='0.0.0.0', port=port_num)