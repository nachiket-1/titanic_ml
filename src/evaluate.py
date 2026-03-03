import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np

from src.preprocess import load_and_prepare, build_preprocessor


DATA_PATH  = "data/titanic.csv"
MODEL_PATH = "models/titanic_model.pkl"


def compare_models(X_train, y_train, preprocessor):
    """Quick cross-val comparison of 3 models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    }
    print("\n📊 Cross-Validation Comparison (5-fold accuracy):")
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  {name:<25} {scores.mean():.4f} ± {scores.std():.4f}")


def tune_random_forest(X_train, y_train, preprocessor):
    """GridSearchCV tuning for Random Forest."""
    print("\n🔍 Tuning Random Forest with GridSearchCV...")

    param_grid = {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [4, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features":  ["sqrt", "log2"],
    }

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(
        pipe, param_grid, cv=5,
        scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"✅ Best CV Accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train():
    print("📂 Loading and preparing data...")
    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    preprocessor = build_preprocessor()

    # Step 1: Compare models
    compare_models(X_train, y_train, preprocessor)

    # Step 2: Tune best model
    best_model = tune_random_forest(X_train, y_train, preprocessor)

    # Step 3: Save model + test data
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":   best_model,
        "X_test":  X_test,
        "y_test":  y_test
    }, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")
    return best_model, X_test, y_test


if __name__ == "__main__":
    train()