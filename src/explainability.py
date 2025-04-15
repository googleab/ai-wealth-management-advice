# src/explainability.py
import shap
import matplotlib.pyplot as plt
import joblib
from src.config import MODEL_SAVE_PATH

def load_model():
    """Load the saved model."""
    return joblib.load(MODEL_SAVE_PATH)

def calculate_shap_values(model, X):
    """Calculate SHAP values using the general SHAP Explainer."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return explainer, shap_values

def plot_global_feature_importance(explainer, X):
    shap_values = explainer.shap_values(X)
    
    # Ensure shap_values is 2D
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)
    
    # Plot the summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    return plt.gcf()


def plot_local_explanation(shap_values, X, instance_index: int = 0):
    """Generate a local explanation force plot for one instance."""
    return shap.force_plot(
        shap_values.base_values[instance_index],
        shap_values.values[instance_index],
        X.iloc[instance_index],
        matplotlib=True
    )
