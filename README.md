# Image Classification using Bag-of-Visual-Words (BoVW) and Machine Learning

This project implements an image classification pipeline using the **Bag-of-Visual-Words (BoVW)** model combined with various classical **machine learning classifiers**, including SVM, Naive Bayes, Logistic Regression, Random Forest, KNN, and MLP.

We extract ORB-based features from grayscale images and compare classification performance across models.

---

## Project Objective

Explore the performance of different machine learning classifiers for image classification using the BoVW framework and identify the most effective model.

---

## Dataset

* Source: Caltech Vision Dataset
* Total images: 2000 (grayscale)
* **Classes**: `Airplanes`, `Faces`, `Motorbikes`
* **Split**:

  * **Training**: 1700 images

    * Airplanes: 700
    * Faces: 300
    * Motorbikes: 700
  * **Testing**: 300 images (100 per class)

---

## Method Overview

### 1. Feature Extraction

* ORB (Oriented FAST and Rotated BRIEF) is used to detect local keypoints.
* ORB descriptors are extracted to represent each keypoint as a binary vector.

### 2. Visual Vocabulary Construction

* All ORB descriptors from the training set are clustered using **KMeans**.
* Cluster centers form the **visual vocabulary** (codebook) of size `k=200`.

### 3. Histogram Representation (BoVW)

* Each image is converted into a histogram of visual word frequencies.
* Resulting BoVW vectors are 200-dimensional and used as input features for classification.

### 4. Classification Models

Six classifiers are trained using the BoVW features:

* **SVM** (Support Vector Machine)
* **NB** (Naive Bayes)
* **LR** (Logistic Regression)
* **RF** (Random Forest)
* **KNN** (K-Nearest Neighbors)
* **MLP** (Multi-Layer Perceptron)

All models use a unified pipeline: scaling → feature selection → TF-IDF → classifier.

---

## 📈 Evaluation & Analysis

* Accuracy is evaluated on both training and test datasets.
* Confusion matrices are generated for each model.
* Feature visualizations (PCA, t-SNE) and accuracy vs. `k` (number of clusters) are analyzed.

---

## Installation

### 0. Prerequisites

Download the dataset from:
🔗 [https://drive.google.com/file/d/1C1uMYMlDDFEbn6BwoDOFO5l6rwuZ\_0tA/view?usp=sharing](https://drive.google.com/file/d/1C1uMYMlDDFEbn6BwoDOFO5l6rwuZ_0tA/view?usp=sharing)
Unzip into the `data/` directory.

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. (Optional) Development dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Usage

### 1. Train Models

```bash
python pipeline_train.py
```

This will extract features, train all six models, and save results in:

* `models/` (pipelines and codebooks)
* `results/` (confusion matrices, plots, metrics)

### 2. Evaluate on Test Set

```bash
python eval_pip.py
```

### 3. Visualize & Analyze Features

```bash
python analysis.py
```

Generates:

* ORB keypoints visualizations
* PCA & t-SNE feature plots
* Accuracy vs. `k` plots for BoVW vocabulary size

---

## Sample Results (Test Accuracy)

| Model | Test Accuracy |
| ----- | ------------- |
| SVM   | 92.0%         |
| NB    | 86.7%         |
| LR    | 91.3%         |
| RF    | 89.3%         |
| KNN   | 87.7%         |
| MLP   | **93.7%**     |

---

