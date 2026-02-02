import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data.dataloader import EyeDiseaseDataset
from models.cnn_model import EyeDiseaseClassifier


def evaluate():
    DATA_DIR = "dataset"
    BATCH_SIZE = 32
    MODEL_PATH = "checkpoints/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = EyeDiseaseDataset(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    _, val_loader, class_names = dataset.get_dataloaders()

    # Model
    model = EyeDiseaseClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

    # Save results
    results = {
        "accuracy": acc,
        "report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = "results/confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    evaluate()

