# src/model.py

import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os
from src.data_processing import load_and_preprocess_data
from src.config import DATA_PATH, MODEL_SAVE_PATH

def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, preds)
    return score

def train_and_tune_model():
    X_processed, y, pipeline = load_and_preprocess_data(DATA_PATH)
    X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=50)
    
    best_params = study.best_trial.params
    best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42})
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    auc_score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"Optimized ROC-AUC Score: {auc_score:.4f}")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model

if __name__ == '__main__':
    train_and_tune_model()

print("Saving model to:", os.path.dirname(MODEL_SAVE_PATH))
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
