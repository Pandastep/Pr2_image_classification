import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from helper import extract_feature
from preprocess import get_training_data
from config import config
from features import build_histogram

LABELS = ['airplanes', 'faces', 'motorbikes']


def visualize_orb_keypoints(image_path, save_path):
    print("[ORB] Visualizing keypoints...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    img_kp = cv2.drawKeypoints(img, kp, None, flags=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title("ORB Keypoints")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[ORB] Saved to {save_path}")


def visualize_bovw(features, labels, method='tsne', save_path="results/plots/bovw_tsne.png"):
    print(f"[BOVW] Reducing dimensions using {method.upper()}...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(features)
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    class_names = le.classes_

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for class_index, class_name in enumerate(class_names):
        mask = numeric_labels == class_index
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=class_name, alpha=0.6, s=30)

    plt.title(f"{method.upper()} of BoVW Features")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[BOVW] {method.upper()} visualization saved to {save_path}")


def k_vs_accuracy():
    print("[K-TEST] Running accuracy vs k analysis...")
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score

    k_values = [50, 100, 200, 300, 400]
    accuracies = []

    image_paths, labels = get_training_data()
    orb = cv2.ORB_create()
    descriptors_list = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des)

    all_descriptors = np.vstack(descriptors_list)

    for k in k_values:
        print(f"[K-TEST] Clustering with k = {k}...")
        kmeans = KMeans(n_clusters=k, random_state=42).fit(all_descriptors)
        histograms = []
        valid_labels = []

        for i, path in enumerate(image_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = orb.detectAndCompute(img, None)
            if des is not None:
                hist = build_histogram(des, kmeans.cluster_centers_, k)
                histograms.append(hist)
                valid_labels.append(labels[i])

        X = np.array(histograms)
        y = np.array(valid_labels)
        model = SVC()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        accuracies.append(acc)
        print(f"[K-TEST] k = {k} | Accuracy = {acc:.4f}")

    os.makedirs("results/plots", exist_ok=True)
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Accuracy vs k (BoVW)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Training Accuracy")
    plt.grid(True)
    plt.savefig("results/plots/accuracy_vs_k.png", bbox_inches='tight')
    plt.close()
    print("[K-TEST] Plot saved to results/plots/accuracy_vs_k.png")


if __name__ == "__main__":
    print("[START] Running analysis...")
    visualize_orb_keypoints("data/training/faces/image_0001.jpg", "results/plots/orb_keypoints.png")

    print("Loading features for t-SNE and PCA...")
    images, labels = get_training_data()
    img_features_raw, _ = extract_feature(images)
    selector = joblib.load("models/svm_pipeline.pkl").named_steps['selector']
    tfidf = joblib.load("models/svm_pipeline.pkl").named_steps['tfidf']

    img_features = selector.transform(img_features_raw)
    img_features = tfidf.transform(img_features).toarray()

    visualize_bovw(img_features, labels, method='tsne', save_path="results/plots/bovw_tsne.png")
    visualize_bovw(img_features, labels, method='pca', save_path="results/plots/bovw_pca.png")

    k_vs_accuracy()
    print("[DONE] Analysis complete.")
