import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.cnn_model import EyeDiseaseClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

model = EyeDiseaseClassifier(num_classes=4)
model.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=DEVICE)
)
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(probs[0][i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }
