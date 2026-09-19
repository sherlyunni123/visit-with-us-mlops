import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

#configuration
TRAIN_FILE = "tourism_project/model_building/Xtrain.csv"
TEST_FILE = "tourism_project/model_building/Xtest.csv"
YTRAIN_FILE = "tourism_project/model_building/ytrain.csv"
YTEST_FILE = "tourism_project/model_building/ytest.csv"

MODEL_DIR = "tourism_project/deployment"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")

#TARGET = "ProdTaken"
#load train and test data
print(f"Loading training data from: {TRAIN_FILE}")
X_train = pd.read_csv(TRAIN_FILE)

print(f"Loading testing data from: {TEST_FILE}")
X_test = pd.read_csv(TEST_FILE)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

#target
print(f"Loading target training data from: {YTRAIN_FILE}")
y_train = pd.read_csv(YTRAIN_FILE).squeeze()

print(f"Loading target testing data from: {YTEST_FILE}")
y_test = pd.read_csv(YTEST_FILE).squeeze()

print(f"Training target data shape: {y_train.shape}")
print(f"Testing target data shape: {y_test.shape}")

#identify numerical and categorical columns
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

#preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

#define model
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

#create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

#hyperparameter grid
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}
#MLFlow experiment
mlflow.set_experiment("Visit_with_Us_Wellness_Tourism")
with mlflow.start_run():
  print("\nStarting hyperparameter tuning...")
  grid_search = GridSearchCV(
      estimator=pipeline,
      param_grid=param_grid,
      cv=5,
      scoring="f1",
      n_jobs=-1
  )
  grid_search.fit(X_train, y_train)
  best_model = grid_search.best_estimator_
  print("\nBest parameters:")
  print(grid_search.best_params_)

  #Predictions
  y_pred = best_model.predict(X_test)
  y_prob = best_model.predict_proba(X_test)[:, 1]

  #Evaluation metrics
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, zero_division=0)
  recall = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)
  roc_auc = roc_auc_score(y_test, y_prob)

  #Log parameters
  for parameter, value in grid_search.best_params_.items():
    mlflow.log_param(parameter, value)

  #log metrics
  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("f1_score", f1)
  mlflow.log_metric("roc_auc", roc_auc)

  #Log model to MLflow
  mlflow.sklearn.log_model(
      best_model,
      "model"
  )

#print evaluation results
  print("\nBest Model Evaluation")
  print(f"Accuracy : {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall   : {recall:.4f}")
  print(f"F1 Score : {f1:.4f}")
  print(f"ROC-AUC  : {roc_auc:.4f}")

#create deployment directory
os.makedirs(MODEL_DIR, exist_ok=True)

#save best model
joblib.dump(best_model, MODEL_FILE)

print(f"\nBest model saved to: {MODEL_FILE}")

