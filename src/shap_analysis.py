import pandas as pd
from src.explainability import (
    load_model, calculate_shap_values,
    plot_global_feature_importance, plot_local_explanation
)
from src.data_processing import load_and_preprocess_data

# Load and preprocess data
X, y, pipeline = load_and_preprocess_data("data/client_data.csv")

# Load model
model = load_model()

# Calculate SHAP values
explainer, shap_values = calculate_shap_values(model, X)

# Global feature importance plot
plot = plot_global_feature_importance(explainer, X)
plot.savefig("dashboards/global_shap_summary.png")  # Save plot to file

# Local explanation
plot_local_explanation(explainer, X, instance_index=0)
