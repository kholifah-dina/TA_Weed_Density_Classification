import os
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import preprocess_image
from feature_extraction import extract_features, select_features

def train_models(dataset):
    """
    dataset is a dictionary: {'Renggang': [img_bytes, ...], 'Sedang': [], 'Padat': []}
    """
    X_all = []
    y_all = []

    # 1. Feature Extraction pipeline
    for class_name, img_bytes_list in dataset.items():
        for img_bytes in img_bytes_list:
            # Preprocess
            clear_img = preprocess_image(image_bytes=img_bytes)
            # Extract features
            features = extract_features(clear_img)
            # Apply feature selection directly
            selected = select_features(features)
            
            X_all.append(selected)
            y_all.append(class_name)

    X = np.array(X_all)
    y = np.array(y_all)

    # 2. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 3. Models Setup
    models = {
        "logistic_regression": LogisticRegression(solver='lbfgs', random_state=0, C=1, class_weight=None, max_iter=300, penalty='l2'),
        "svm": SVC(probability=True, random_state=0, C=5, degree=2, gamma=0.01, kernel='rbf'),
        "decision_tree": DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=3, max_leaf_nodes=4, min_samples_leaf=3, min_samples_split=3),
        "random_forest": RandomForestClassifier(random_state=0, n_estimators=500),
        "gradient_boosting": GradientBoostingClassifier(random_state=0, n_estimators=300, learning_rate=0.1)
    }

    trained_models = {}
    evaluation_metrics = {}
    confusion_matrices = {}

    labels = ["Sedang", "Padat", "Renggang"]
    # Provide consistent ordering for metrics
    unique_labels = sorted(list(set(y_all)))

    # 4. Training and Evaluation
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        evaluation_metrics[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        }
        
        confusion_matrices[model_name] = {
            "matrix": cm,
            "labels": unique_labels
        }
        
        trained_models[model_name] = model

    # 5. Save all necessary pipeline info
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Store models in the exact dictionary format requested
    joblib.dump(trained_models, "models/weed_models.joblib")

    # Store metrics separately to keep model file clean
    saved_metrics = {
        "metrics": evaluation_metrics,
        "confusion_matrices": confusion_matrices,
        "features": ['homogeneity_90deg', 'homogeneity_45deg', 'energy_45deg', 'energy_135deg',
                     'energy_90deg','energy_0deg','homogeneity_0deg','homogeneity_135deg',
                     'dissimilarity_90deg','dissimilarity_135deg','dissimilarity_45deg',
                     'dissimilarity_0deg','HuMoment_6','G_mean']
    }
    joblib.dump(saved_metrics, "models/weed_metrics.joblib")
    
    return saved_metrics

def test_inference(image_bytes):
    """
    Run the trained model on a single new image and return predictions
    """
    trained_models = joblib.load('models/weed_models.joblib')
    saved_metrics = joblib.load('models/weed_metrics.joblib')
    
    # Preprocess and extract
    clear_img = preprocess_image(image_bytes=image_bytes)
    features = extract_features(clear_img)
    selected = select_features(features)
    
    X_single = np.array([selected])
    
    predictions = {}
    for model_name, model in trained_models.items():
        predictions[model_name] = model.predict(X_single)[0]
        
    return predictions, saved_metrics
