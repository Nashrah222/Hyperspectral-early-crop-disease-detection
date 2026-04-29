# AgroAgent: Fast Crop Disease Detection 🌱

AgroAgent is an optimized, end-to-end machine learning pipeline designed for the rapid detection and classification of crop diseases using hyperspectral reflectance data. 


By leveraging a deep 1D Convolutional Neural Network (CNN) combined with spatial smoothing (Median Filtering) and morphological processing, the system accurately classifies spectral signatures and visualizes stressed/diseased regions on a pixel-by-pixel basis.

## 🚀 Key Features

* **End-to-End Pipeline:** Automatically handles everything from raw `.hdr` file parsing to model training and final dashboard generation.
* **1D CNN Architecture:** Uses a custom deep learning model specifically designed to extract features from 1D spectral arrays.
* **Spatial Smoothing:** Implements a 7x7 Median Filter to eliminate spectral noise, smoothing out pixel-wise predictions.
* **Smart Subsampling:** Trains on a balanced 20% subset of the data to drastically reduce training time while maintaining baseline accuracy.
* **Automated Bounding Boxes:** Uses morphological opening/closing and connected component analysis to isolate and draw bounding boxes around major disease clusters.
* **Instant Dashboard:** Generates a comprehensive visual report including a Confusion Matrix, a Pixel-wise Stress Map, and Final Disease Detection overlays.

## 📊 Model Performance

Based on the latest evaluation report, the model achieves an overall pixel-wise classification accuracy of **63%** across four classes. 

**Detailed Classification Report:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Background  | 0.94      | 0.76   | 0.84     | 224,308 |
| Healthy     | 0.46      | 0.23   | 0.31     | 84,175  |
| Stressed    | 0.51      | 0.55   | 0.53     | 127,700 |
| Diseased    | 0.50      | 0.75   | 0.60     | 150,455 |
| **Accuracy**|           |        | **0.63** | 586,638 |
| Macro Avg   | 0.60      | 0.57   | 0.57     | 586,638 |
| Weighted Avg| 0.66      | 0.63   | 0.63     | 586,638 |

## 📁 Project Structure

```text
CROP_DISEASE_DETECTION/
│
├── data/                         # Raw .hdr files, cleaned .npy data, and labels
├── models/
│   ├── saved_models/             # Directory for best_model.h5
│   └── cnn_model.py              # 1D CNN architecture and training logic
│
├── object_classification/        
│   └── region_props.py           # Logic for region bounding and object detection
│
├── preprocessing/                
│   ├── load_data.py              # HDR ingestion and parsing
│   └── noise_removal.py          # Data cleaning and noise reduction
│
├── results/                      
│   ├── class_analysis.py         
│   ├── evaluate.py               # Evaluation scripts
│   ├── generate_map.py           
│   ├── confusion_matrix.png      # Generated output visual
│   ├── final_dashboard.png       # Generated output visual
│   ├── final_detection.png       # Generated output visual
│   ├── final_detection_clean.png # Generated output visual
│   ├── stress_map.png            # Generated output visual
│   └── performance_report.txt    # Text metrics (Precision/Recall/F1)
│
├── fast_main.py                  # Optimized main execution pipeline (subsampled/smoothed)
└── main.py                       # Standard execution pipeline
