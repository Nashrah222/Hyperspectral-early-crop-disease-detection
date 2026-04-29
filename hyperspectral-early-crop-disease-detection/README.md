# Hyperspectral Early Crop Disease Detection using HybridSN

![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Python](https://img.shields.io/badge/Language-Python-green)
![Computer Vision](https://img.shields.io/badge/Domain-ComputerVision-red)
![Status](https://img.shields.io/badge/Status-Research-success)

A deep learning pipeline that detects crop disease at **pixel level** using hyperspectral imaging data.
The system processes raw ENVI `.hdr` spectral cubes and classifies each pixel into **Healthy, Stressed, or Diseased** categories using a **HybridSN architecture (3D CNN + 2D CNN)** achieving **92.8% validation accuracy**.

---

# Project Overview

Traditional RGB crop monitoring fails to detect early-stage disease because visible symptoms appear only after significant plant damage.

Hyperspectral imaging captures hundreds of spectral bands including near-infrared wavelengths, enabling detection of biochemical stress signals such as:

* chlorophyll degradation
* water content changes
* pigment variation

This project builds an end-to-end AI pipeline that converts raw hyperspectral data into disease maps and severity dashboards.

---

# Key Features

* Pixel-level crop disease classification
* HybridSN architecture combining 3D CNN and 2D CNN
* PCA-based dimensionality reduction for hyperspectral bands
* Spectral derivative feature extraction using Savitzky-Golay filtering
* Spatially-aware train-test split to prevent data leakage
* Confidence heatmap for prediction reliability
* Bounding box detection for infected crop regions
* Synthetic data generation for robustness testing
* Automated 6-step pipeline from raw data to disease visualization

---

# Tech Stack

### AI / ML

* TensorFlow / Keras
* HybridSN (3D CNN + 2D CNN)
* Squeeze-and-Excitation Attention block
* PCA (Principal Component Analysis)
* Savitzky-Golay spectral derivatives

### Data Processing

* NumPy
* SciPy
* scikit-learn
* scikit-image

### Visualization

* Matplotlib
* Seaborn

### Data Format

* ENVI hyperspectral format (.hdr)
* NumPy arrays (.npy)
* Pickle serialization (.pkl)

### Environment

* Python 3.10+
* CUDA GPU acceleration

---

# System Pipeline

Raw hyperspectral cube (.hdr)

↓

Spectral preprocessing

↓

Dimensionality reduction using PCA

↓

7×7 spatial patch extraction

↓

HybridSN deep learning model training

↓

Pixel-wise disease classification

↓

Confidence heatmap generation

↓

Disease region detection using morphological operations

↓

Final disease visualization dashboard

---

# Architecture

HybridSN model structure:

3D Convolution layers learn joint spectral-spatial features

↓

Feature reshaping

↓

2D Convolution layers refine spatial patterns

↓

Squeeze-and-Excitation block enhances important channels

↓

Dense classifier outputs probability scores

---

# Model Performance

Validation Accuracy: 92.8%

Classes:

* Healthy
* Stressed
* Diseased

Evaluation outputs:

* confusion matrix
* classification report
* confidence heatmap
* disease detection map

---

# Important Functionalities

### Triple-channel spectral feature extraction

Combines:

* raw spectrum
* first derivative
* second derivative

Improves detection of plant biochemical stress signals.

### Leakage-free spatial train-test split

Training and test pixels are separated spatially to prevent overlapping patches.

### Memory-efficient training

Patch batches streamed dynamically to GPU to reduce memory usage.

### Chunk-wise inference

Processes large hyperspectral images without RAM overflow.

### Severity grading

Detected disease regions categorized into:

* Low
* Moderate
* High
* Critical

based on confidence scores.

---

# Setup Instructions

## Requirements

Python 3.10+

TensorFlow 2.11

NumPy

SciPy

scikit-learn

scikit-image

Matplotlib

Seaborn

spectral library

CUDA (optional for GPU)

---

## Installation

Clone repository

git clone https://github.com/your-username/hyperspectral-disease-detection.git

cd hyperspectral-disease-detection

Create virtual environment

python -m venv gpu_env

Activate environment

Windows:
gpu_env\Scripts\activate

Mac/Linux:
source gpu_env/bin/activate

Install dependencies

pip install tensorflow numpy scipy scikit-learn scikit-image matplotlib seaborn spectral

---

## Run Pipeline

python main.py

Pipeline automatically executes:

1. Data loading
2. PCA preprocessing
3. Patch extraction
4. Model training
5. Evaluation
6. Disease map generation

---

# Future Improvements

* real-time drone-based hyperspectral detection
* mobile dashboard for farmers
* multi-crop dataset expansion
* improved disease severity prediction
* integration with IoT farm sensors

---

# Applications

precision agriculture

early disease detection

crop monitoring

smart farming systems

yield optimization

agricultural AI research

---
