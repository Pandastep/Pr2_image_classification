import os
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from helper import extract_feature, to_dense
from preprocess import get_training_data

LABELS = ['airplanes', 'faces', 'motorbikes']

def save_training_plot(model_name, train_time, predict_time, accuracy, save_time):
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Time data for plotting
    times = [train_time, predict_time, save_time]
    labels = ['Training', 'Prediction', 'Saving']

    # Create figure
    plt.figure(figsize=(10, 6))

    # Time distribution plot
    plt.subplot(1, 2, 1)
    plt.bar(labels, times)
    plt.title(f'{model_name} Time Distribution')
    plt.ylabel('Time (seconds)')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [accuracy])
    plt.title(f'{model_name} Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_training_results.png')
    plt.close()

def train_model(model_name, classifier):
    print(f"\n=== Training {model_name} ===")
    start_total = time.time()

    print("Loading training data...")
    image_paths, labels = get_training_data()
    print(f"Number of training images: {len(image_paths)}")

    print("Extracting features...")
    features, codebook = extract_feature(image_paths)
    print(f"Feature shape: {features.shape}")

    print("Building pipeline...")
    if model_name == "nb":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=100)),
            ('tfidf', TfidfTransformer()),
            ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
            ('clf', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=100)),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)
        ])

    print("Training pipeline...")
    start_train = time.time()
    pipeline.fit(features, labels)
    train_time = time.time() - start_train

    print("Predicting on training data...")
    preds = pipeline.predict(features)
    acc = accuracy_score(labels, preds)
    predict_time = time.time() - start_train - train_time

    print(f"Training time: {train_time:.2f}s")
    print(f"Prediction time: {predict_time:.2f}s")
    print(f"Training accuracy: {acc:.4f}")

    print("Saving model and artifacts...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, f"models/{model_name}_pipeline.pkl")
    joblib.dump(codebook, f"models/{model_name}_codebook.pkl")

    print("Saving confusion matrix...")
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap='PuBu')
    plt.title(f"{model_name} Confusion Matrix")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_train_confusion.png")
    plt.close()

    save_time = time.time() - (start_total + train_time + predict_time)
    print("Saving training summary plot...")
    save_training_plot(model_name, train_time, predict_time, acc, save_time)

    print(f"Model and artifacts for {model_name} saved.")
    print(f"Total time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    train_model("svm", SVC(max_iter=10000, random_state=42))
    train_model("nb", GaussianNB())
    train_model("logreg", LogisticRegression(max_iter=1000, random_state=42))
    train_model("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    train_model("knn", KNeighborsClassifier(n_neighbors=5))
    train_model("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
