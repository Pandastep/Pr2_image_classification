import os
import time
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from preprocess import get_test_data
from features import build_histogram
from config import config

LABELS = ['airplanes', 'faces', 'motorbikes']


def evaluate_pipeline(model_name: str):
    print(f"\n=== Evaluating {model_name} on Test Set ===")
    start_total = time.time()

    pipeline_path = f"models/{model_name}_pipeline.pkl"
    codebook_path = f"models/{model_name}_codebook.pkl"

    if not os.path.exists(pipeline_path) or not os.path.exists(codebook_path):
        print(f"Missing model or codebook for {model_name}. Skipping.")
        return

    print("Loading model and codebook...")
    model = joblib.load(pipeline_path)
    codebook = joblib.load(codebook_path)
    k = config.CLUSTER_SIZE

    print("Loading test data...")
    test_images, test_labels = get_test_data()
    print(f"Number of test images: {len(test_images)}")

    orb = cv2.ORB_create()
    histograms = []
    valid_labels = []

    print("Building histograms from test data...")
    for i, path in enumerate(test_images):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            hist = build_histogram(descriptors, codebook, k)
            histograms.append(hist)
            valid_labels.append(test_labels[i])

    if not histograms:
        print("No valid histograms generated. Evaluation aborted.")
        return

    X_test = np.array(histograms)
    y_test = np.array(valid_labels)

    print("Making predictions...")
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    acc = accuracy_score(y_test, y_pred)
    print(f"[{model_name.upper()}] Test Accuracy: {acc:.4f}")
    print(f"Prediction Time: {pred_time:.2f}s")
    print(f"Total Evaluation Time: {time.time() - start_total:.2f}s")

    # Confusion matrix
    print("Saving confusion matrix plot...")
    os.makedirs("results/test_confusions", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap='YlOrBr')
    plt.title(f"{model_name.upper()} Confusion Matrix (Test)")
    plt.savefig(f"results/test_confusions/{model_name}_test_confusion.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("[START] Evaluating all models on test set...")
    for model in ["svm", "nb", "logreg", "rf", "knn", "mlp"]:
        evaluate_pipeline(model)
    print("[DONE] All evaluations complete.")