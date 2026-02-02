# AI Eye Disease Screener 👁️

An AI-based image classification system for screening eye diseases such as:
- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- Albumentations
- FastAPI (deployment-ready)

---

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/Nabeel-Aly/ai-eye-diseases-screening
cd ai-eye-disease-screener
````

### 2. Create Virtual Environment

**Windows**

```bash
python -m venv venv
```

**Linux / macOS**

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

### Dataset Structure

```text
dataset/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/
```

### Data Loading & Preprocessing

- Uses PyTorch `ImageFolder`
- Augmentations via Albumentations
- Automatic train/validation split
- Image size: 224×224
- Normalized using ImageNet statistics

Data loaders are defined in:
```bash
src/data/dataloader.py
```

---

## Model Architecture

- Backbone: ResNet18 (pretrained on ImageNet)
- Custom classification head
- Output classes: 4
- Transfer learning enabled

Model definition:
```bash
src/models/cnn_model.py
```

---

## Training

Run training with:
```bash
python src/train.py
```

- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **LR Scheduler:** ReduceLROnPlateau
- Checkpoints saved in checkpoints/

---

## Evaluation

Run evaluation:
```bash
python src/evaluate.py
``` 

### Metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix
- Evaluation outputs:
- Terminal classification report
- Confusion matrix visualization

---

## FastAPI Deployment

Run the web app:
```bash
uvicorn src.api.main:app
```

Reload the web app:
```bash
uvicorn src.api.main:app --reload
```

**Open:** 
http://127.0.0.1:8000

Upload an eye image to receive AI-based screening results.

---