import shap
import numpy as np

def feature_importance_method(X, y, feature_names, trained_model):
    explainer = shap.Explainer(trained_model, X)
    shap_values = explainer(X)
    shap_mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    feature_indices = np.argsort(shap_mean_abs)[::-1]
    sorted_importances = [(feature_names[i], shap_mean_abs[i]) for i in feature_indices]
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar')
    return sorted_importances
