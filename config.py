import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}