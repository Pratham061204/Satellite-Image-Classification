# Satellite-Image-Classification

This repository implements a **Satellite Image Classification** pipeline using the **UC Merced Land Use Dataset**—a collection of 2,100 aerial images across 21 land-use classes. Leveraging **Transfer Learning** with pretrained CNNs with custom layers and a custom CNN(for baseline purposes), the project achieves high accuracy ~98%) and provides a Flask web interface for real-time predictions.

---
## Project Overview
This project explores classification of satellite imagery using the UC Merced Land Use Dataset. Key objectives:

- **Data**: 21 land-use categories (e.g., agricultural, residential, forest). 100 images per class (256×256 px color).  
- **Transfer Learning**: Fine-tune **MobileNet** and **VGG16** backbones; achieved **98.3%** validation accuracy with MobileNet.  
- **Custom CNN**: Baseline architecture trained from scratch to compare performance.  
- **Augmentation**: Random rotations, flips, and shifts to improve robustness.  
- **Flask Interface**: Upload new images and get class predictions instantly.

---
## Libraries & Dependencies
| Library                | Purpose                                              |
|------------------------|------------------------------------------------------|
| `tensorflow` / `keras` | Model building, transfer learning, training loops    |
| `opencv-python`        | Image loading & low-level preprocessing              |
| `scikit-image`         | Dataset I/O (`skimage.io`)                           |
| `Pillow`               | Image file conversions                               |
| `numpy`                | Array operations                                     |
| `matplotlib`           | Plotting training curves                             |
| `scikit-learn`         | Classification report, confusion matrix, metrics     |
| `Flask`                | Web framework for demo interface                     |
| `werkzeug`             | Secure filenames for uploads                         |

## Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/satellite-image-classification.git
cd satellite-image-classification

python3 -m venv venv
source venv/bin/activate  # for Mac/Linux
venv\Scripts\activate    # for Windows

pip install -r requirements.txt
```
## Data Preparation
1. Download the UC Merced Land Use Dataset from the official site.
2. Extract into data/train/{class}/ and data/val/{class}/ (70/15 split).
3. The notebook details preprocessing and augmentation steps.

## Training & Evaluation
Notebook shows how to compile, train, and evaluate models with callbacks (ModelCheckpoint, EarlyStopping).
Run training scripts or execute notebook cells to reproduce results or directly use the saved weights

## Model Architecture
- MobileNet(pretrained on ImageNet) :
  -  frozen base + custom classifier head
    - Dense(4096,activation='relu')
    - Dense(1072,activation='relu')
    - Dropout(0.5)
    - Dense(21,activation='softmax').

- VGG16: similar setup for comparison.

- Custom CNN(baseline)

## Results
MobileNet performs the best 98% accuracy on test set compared 96% of VGG16 and 72% Custom CNN 

## Flask Web App
1) Navigate to http://127.0.0.1:5000
2) Upload a satellite image
3) View the predicted land-use class
Use load_model() to load your chosen model and preprocess input for inference.


