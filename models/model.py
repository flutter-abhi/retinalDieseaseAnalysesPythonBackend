import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn.functional as F

class OcularDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(OcularDiseaseModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get the number of features
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Create new classifier with batch norm and dropout
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
            # Note: No sigmoid here as we're using BCE with logits
        )
        
    def forward(self, left_eye, right_eye):
        # Process both eyes
        left_features = self.backbone(left_eye)
        right_features = self.backbone(right_eye)
        
        # Get logits for each eye
        left_logits = self.classifier(left_features)
        right_logits = self.classifier(right_features)
        
        # Average the logits
        combined_logits = (left_logits + right_logits) / 2
        
        return {
            'left': left_logits,
            'right': right_logits,
            'combined': combined_logits
        } 