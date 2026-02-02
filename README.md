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
