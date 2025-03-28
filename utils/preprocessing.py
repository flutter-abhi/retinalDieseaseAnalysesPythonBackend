try:
    import numpy as np
except ImportError:
    raise ImportError("Numpy is required. Please install it using: pip install numpy==1.24.3")

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

def preprocess_image(image):
    """Preprocess the uploaded image"""
    try:
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        transformed = transform(image=image_np)
        return transformed['image'].unsqueeze(0)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise