"""
Model Evaluation Module
Comprehensive model evaluation with metrics, visualizations, and explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

from utils.helpers import (
    show_learning_tip, create_confusion_matrix_plot, create_feature_importance_plot,
    create_metrics_comparison_chart
)

def show():
    """Main function to display the model evaluation interface"""
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        return
    
    st.title("📈 Model Evaluation Module")
    st.markdown("Comprehensive evaluation and analysis of your trained model!")
    
    # Learning tip
    show_learning_tip(
        "Model evaluation goes beyond just accuracy. Understanding different metrics, "
        "visualizations, and model behavior helps you make informed decisions about model deployment.",
        "info"
    )
    
    # Evaluation sections
    eval_tabs = st.tabs([
        "📊 Performance Metrics",
        "🎯 Confusion Matrix", 
        "📈 ROC & Precision-Recall",
        "🔍 Feature Analysis",
        "🧠 Model Interpretation",
        "📋 Detailed Report"
    ])
    
    with eval_tabs[0]:
        show_performance_metrics()
    
    with eval_tabs[1]:
        show_confusion_matrix()
    
    with eval_tabs[2]:
        show_roc_and_pr_curves()
    
    with eval_tabs[3]:
        show_feature_analysis()
    
    with eval_tabs[4]:
        show_model_interpretation()
    
    with eval_tabs[5]:
        show_detailed_report()

def show_performance_metrics():
    """Display comprehensive performance metrics"""
    
    st.markdown("## 📊 Performance Metrics")
    st.markdown("Detailed analysis of your model's performance")
    
    if st.session_state.metrics is None:
        st.warning("⚠️ No metrics available. Please retrain your model.")
        return
    
    metrics = st.session_state.metrics
    model_name = st.session_state.model_name
    
    # Overview metrics
    st.markdown(f"### 🎯 {model_name} Performance Overview")
    
    # Main metrics in columns
    metric_cols = st.columns(len(metrics))
    
    for i, (metric, value) in enumerate(metrics.items()):
        with metric_cols[i]:
            # Color coding based on performance
            if value >= 0.9:
                delta_color = "normal"
            elif value >= 0.8:
                delta_color = "normal"
            else:
                delta_color = "inverse"
            
            st.metric(
                label=metric,
                value=f"{value:.3f}",
                delta=get_metric_interpretation(metric, value)
            )
    
    # Metrics radar chart
    st.markdown("### 🕸️ Performance Radar Chart")
    fig = create_metrics_comparison_chart(metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics explanation
    st.markdown("### 📚 Metrics Explanation")
    
    with st.expander("🔍 What do these metrics mean?"):
        explain_metrics(metrics)
    
    # Performance interpretation
    st.markdown("### 🎯 Performance Interpretation")
    interpret_model_performance(metrics, model_name)
    
    # Comparison with baselines
    show_baseline_comparison(metrics)

def get_metric_interpretation(metric: str, value: float) -> str:
    """Get interpretation for metric values"""
    
    if metric == "Accuracy":
        if value >= 0.95:
            return "Excellent"
        elif value >= 0.85:
            return "Good"
        elif value >= 0.75:
            return "Fair"
        else:
            return "Needs Improvement"
    
    elif metric in ["Precision", "Recall", "F1-Score"]:
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.80:
            return "Good"
        elif value >= 0.70:
            return "Fair"
        else:
            return "Poor"
    
    elif metric == "ROC-AUC":
        if value >= 0.95:
            return "Outstanding"
        elif value >= 0.85:
            return "Excellent"
        elif value >= 0.75:
            return "Good"
        elif value >= 0.65:
            return "Fair"
        else:
            return "Poor"
    
    return ""

def explain_metrics(metrics: dict):
    """Explain what each metric means"""
    
    explanations = {
        "Accuracy": """
        **Accuracy** measures the percentage of correct predictions out of all predictions.
        - Formula: (TP + TN) / (TP + TN + FP + FN)
        - Best for: Balanced datasets
        - Limitation: Can be misleading with imbalanced data
        """,
        
        "Precision": """
        **Precision** measures how many of the positive predictions were actually correct.
        - Formula: TP / (TP + FP)
        - Interpretation: "Of all items predicted as positive, how many were truly positive?"
        - Important when: False positives are costly (e.g., spam detection)
        """,
        
        "Recall": """
        **Recall (Sensitivity)** measures how many actual positive cases were correctly identified.
        - Formula: TP / (TP + FN)
        - Interpretation: "Of all actual positive items, how many did we catch?"
        - Important when: False negatives are costly (e.g., disease diagnosis)
        """,
        
        "F1-Score": """
        **F1-Score** is the harmonic mean of precision and recall.
        - Formula: 2 × (Precision × Recall) / (Precision + Recall)
        - Interpretation: Balanced measure of precision and recall
        - Best for: When you need a balance between precision and recall
        """,
        
        "ROC-AUC": """
        **ROC-AUC** measures the area under the Receiver Operating Characteristic curve.
        - Range: 0.5 (random) to 1.0 (perfect)
        - Interpretation: Probability that model ranks positive instance higher than negative
        - Best for: Binary classification with balanced classes
        """
    }
    
    for metric, value in metrics.items():
        if metric in explanations:
            st.markdown(f"#### {metric}: {value:.3f}")
            st.markdown(explanations[metric])
            st.markdown("---")

def interpret_model_performance(metrics: dict, model_name: str):
    """Provide interpretation of overall model performance"""
    
    accuracy = metrics.get('Accuracy', 0)
    precision = metrics.get('Precision', 0)
    recall = metrics.get('Recall', 0)
    f1 = metrics.get('F1-Score', 0)
    
    # Overall assessment
    avg_score = np.mean([accuracy, precision, recall, f1])
    
    if avg_score >= 0.9:
        performance_level = "🎉 Excellent"
        color = "success"
    elif avg_score >= 0.8:
        performance_level = "✅ Good"
        color = "success"
    elif avg_score >= 0.7:
        performance_level = "⚠️ Fair"
        color = "warning"
    else:
        performance_level = "❌ Needs Improvement"
        color = "error"
    
    if color == "success":
        st.success(f"**Overall Performance**: {performance_level} (Average: {avg_score:.3f})")
    elif color == "warning":
        st.warning(f"**Overall Performance**: {performance_level} (Average: {avg_score:.3f})")
    else:
        st.error(f"**Overall Performance**: {performance_level} (Average: {avg_score:.3f})")
    
    # Specific insights
    insights = []
    
    # Precision vs Recall analysis
    if precision > recall + 0.1:
        insights.append("🎯 High precision suggests your model is conservative - when it predicts positive, it's usually right!")
    elif recall > precision + 0.1:
        insights.append("🔍 High recall suggests your model catches most positive cases but may have some false alarms.")
    elif abs(precision - recall) < 0.05:
        insights.append("⚖️ Balanced precision and recall indicates good overall classification performance.")
    
    # Model-specific insights
    model_insights = get_model_specific_insights(model_name, metrics)
    insights.extend(model_insights)
    
    if insights:
        st.markdown("#### 💡 Key Insights")
        for insight in insights:
            st.write(f"• {insight}")

def get_model_specific_insights(model_name: str, metrics: dict) -> list:
    """Get model-specific performance insights"""
    
    insights = []
    accuracy = metrics.get('Accuracy', 0)
    
    if model_name == "Logistic Regression":
        if accuracy < 0.7:
            insights.append("Consider feature engineering or polynomial features for better linear separation.")
        else:
            insights.append("Linear model performing well - features have good linear relationship with target.")
    
    elif model_name == "Decision Tree":
        if accuracy > 0.95:
            insights.append("Very high accuracy might indicate overfitting. Consider pruning or cross-validation.")
        elif accuracy < 0.75:
            insights.append("Try adjusting max_depth or min_samples_split to improve performance.")
    
    elif model_name == "Random Forest":
        if accuracy > 0.9:
            insights.append("Excellent performance! Random Forest is handling your data complexity well.")
        else:
            insights.append("Try increasing n_estimators or tuning max_features for better performance.")
    
    elif model_name == "Support Vector Machine":
        if accuracy < 0.8:
            insights.append("SVM might benefit from different kernel or feature scaling.")
        else:
            insights.append("SVM found good decision boundary for your data.")
    
    elif model_name == "K-Nearest Neighbors":
        if accuracy < 0.75:
            insights.append("Try different k values or feature scaling to improve KNN performance.")
        else:
            insights.append("Local patterns in your data are well-captured by KNN.")
    
    elif model_name == "Naive Bayes":
        if accuracy > 0.8:
            insights.append("Feature independence assumption works well for your data.")
        else:
            insights.append("Consider feature selection - some features might violate independence assumption.")
    
    return insights

def show_baseline_comparison(metrics: dict):
    """Show comparison with baseline models"""
    
    st.markdown("### 📊 Baseline Comparison")
    
    # Calculate baselines
    y_test = st.session_state.y_test
    
    # Majority class baseline
    majority_class = y_test.mode().iloc[0] if len(y_test.mode()) > 0 else y_test.iloc[0]
    majority_baseline = (y_test == majority_class).mean()
    
    # Random baseline
    random_baseline = 1.0 / len(y_test.unique())
    
    # Comparison data
    comparison_data = {
        "Model": ["Random Baseline", "Majority Class", f"Your {st.session_state.model_name}"],
        "Accuracy": [random_baseline, majority_baseline, metrics.get('Accuracy', 0)]
    }
    
    fig = px.bar(
        comparison_data,
        x="Model",
        y="Accuracy",
        title="Model vs Baselines",
        color="Accuracy",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Improvement metrics
    accuracy = metrics.get('Accuracy', 0)
    improvement_over_random = (accuracy - random_baseline) / random_baseline * 100
    improvement_over_majority = (accuracy - majority_baseline) / majority_baseline * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Improvement over Random",
            f"{improvement_over_random:.1f}%",
            delta=f"{accuracy - random_baseline:.3f}"
        )
    
    with col2:
        st.metric(
            "Improvement over Majority",
            f"{improvement_over_majority:.1f}%",
            delta=f"{accuracy - majority_baseline:.3f}"
        )

def show_confusion_matrix():
    """Display confusion matrix analysis"""
    
    st.markdown("## 🎯 Confusion Matrix Analysis")
    st.markdown("Detailed breakdown of prediction accuracy by class")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No predictions available. Please retrain your model.")
        return
    
    y_test = st.session_state.y_test
    y_pred = st.session_state.predictions
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    
    # Interactive confusion matrix
    fig = create_confusion_matrix_plot(cm, labels)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix interpretation
    st.markdown("### 🔍 Confusion Matrix Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Matrix Breakdown")
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        for i, true_label in enumerate(labels):
            st.write(f"**True {true_label}:**")
            for j, pred_label in enumerate(labels):
                count = cm[i, j]
                percent = cm_percent[i, j]
                if i == j:
                    st.write(f"  ✅ Correctly predicted as {pred_label}: {count} ({percent:.1f}%)")
                else:
                    st.write(f"  ❌ Incorrectly predicted as {pred_label}: {count} ({percent:.1f}%)")
            st.write("")
    
    with col2:
        st.markdown("#### 🎯 Per-Class Performance")
        
        # Calculate per-class metrics
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            st.write(f"**Class {label}:**")
            st.write(f"  Precision: {precision:.3f}")
            st.write(f"  Recall: {recall:.3f}")
            st.write(f"  F1-Score: {f1:.3f}")
            st.write("")
    
    # Misclassification analysis
    if len(labels) > 2:
        show_misclassification_analysis(cm, labels)

def show_misclassification_analysis(cm: np.ndarray, labels: list):
    """Analyze common misclassifications"""
    
    st.markdown("### 🔍 Misclassification Analysis")
    
    # Find most common misclassifications
    misclassifications = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'True Class': labels[i],
                    'Predicted Class': labels[j],
                    'Count': cm[i, j],
                    'Percentage': (cm[i, j] / cm[i].sum()) * 100
                })
    
    if misclassifications:
        # Sort by count
        misclassifications.sort(key=lambda x: x['Count'], reverse=True)
        
        # Display top misclassifications
        st.markdown("#### 🔴 Most Common Misclassifications")
        
        misclass_df = pd.DataFrame(misclassifications[:10])  # Top 10
        st.dataframe(misclass_df, use_container_width=True)
        
        # Visualization
        if len(misclassifications) > 1:
            fig = px.bar(
                misclass_df.head(5),
                x='Count',
                y=[f"{row['True Class']} → {row['Predicted Class']}" for _, row in misclass_df.head(5).iterrows()],
                orientation='h',
                title="Top 5 Misclassifications"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_roc_and_pr_curves():
    """Display ROC and Precision-Recall curves"""
    
    st.markdown("## 📈 ROC & Precision-Recall Curves")
    st.markdown("Advanced performance analysis with curve visualizations")
    
    if st.session_state.prediction_probabilities is None:
        st.warning("⚠️ No prediction probabilities available. Some models don't support probability prediction.")
        return
    
    y_test = st.session_state.y_test
    y_pred_proba = st.session_state.prediction_probabilities
    
    # Check if binary classification
    if len(y_test.unique()) == 2:
        show_binary_classification_curves(y_test, y_pred_proba)
    else:
        show_multiclass_classification_curves(y_test, y_pred_proba)

def show_binary_classification_curves(y_test, y_pred_proba):
    """Show ROC and PR curves for binary classification"""
    
    # Convert labels to binary if they're strings
    unique_labels = sorted(y_test.unique())
    y_test_binary = (y_test == unique_labels[1]).astype(int)
    
    # Get probabilities for positive class
    if y_pred_proba.shape[1] == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba[:, 0]
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test_binary, y_scores)
    pr_auc = auc(recall, precision)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'ROC Curve (AUC = {roc_auc:.3f})', f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    )
    
    # ROC Curve
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # PR Curve
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve', line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Threshold analysis
    show_threshold_analysis(y_test_binary, y_scores, roc_thresholds, fpr, tpr, precision, recall, pr_thresholds)

def show_multiclass_classification_curves(y_test, y_pred_proba):
    """Show ROC curves for multiclass classification"""
    
    st.info("📊 Multiclass ROC Analysis")
    
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Get unique classes
    classes = sorted(y_test.unique())
    n_classes = len(classes)
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    fig = go.Figure()
    
    for i in range(n_classes):
        fig.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            mode='lines',
            name=f'Class {classes[i]} (AUC = {roc_auc[i]:.3f})',
            line=dict(width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title='Multiclass ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AUC summary
    st.markdown("#### 📊 Per-Class AUC Scores")
    auc_df = pd.DataFrame({
        'Class': [str(classes[i]) for i in range(n_classes)],
        'AUC Score': [roc_auc[i] for i in range(n_classes)]
    })
    st.dataframe(auc_df, use_container_width=True)

def show_threshold_analysis(y_test, y_scores, roc_thresholds, fpr, tpr, precision, recall, pr_thresholds):
    """Show threshold analysis for binary classification"""
    
    st.markdown("### 🎛️ Threshold Analysis")
    
    # Threshold selector
    threshold = st.slider(
        "Select decision threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the threshold to see how it affects predictions"
    )
    
    # Calculate metrics at selected threshold
    y_pred_threshold = (y_scores >= threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred_threshold)
    prec = precision_score(y_test, y_pred_threshold, zero_division=0)
    rec = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
    
    # Display metrics at threshold
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        st.metric("Precision", f"{prec:.3f}")
    with col3:
        st.metric("Recall", f"{rec:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Threshold vs metrics plot
    thresholds_range = np.arange(0.0, 1.01, 0.01)
    threshold_metrics = []
    
    for t in thresholds_range:
        y_pred_t = (y_scores >= t).astype(int)
        threshold_metrics.append({
            'Threshold': t,
            'Accuracy': accuracy_score(y_test, y_pred_t),
            'Precision': precision_score(y_test, y_pred_t, zero_division=0),
            'Recall': recall_score(y_test, y_pred_t, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_t, zero_division=0)
        })
    
    threshold_df = pd.DataFrame(threshold_metrics)
    
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Scatter(
            x=threshold_df['Threshold'],
            y=threshold_df[metric],
            mode='lines',
            name=metric,
            line=dict(width=2)
        ))
    
    # Add vertical line for current threshold
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Current: {threshold}")
    
    fig.update_layout(
        title='Metrics vs Decision Threshold',
        xaxis_title='Threshold',
        yaxis_title='Metric Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis():
    """Display feature importance and analysis"""
    
    st.markdown("## 🔍 Feature Analysis")
    st.markdown("Understanding which features drive your model's predictions")
    
    if st.session_state.feature_importance is None:
        st.warning("⚠️ Feature importance not available for this model type.")
        return
    
    feature_importance = st.session_state.feature_importance
    feature_names = st.session_state.feature_names
    
    # Feature importance plot
    fig = create_feature_importance_plot(feature_names, feature_importance)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.markdown("### 📋 Feature Importance Rankings")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance,
        'Rank': range(1, len(feature_names) + 1)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_df['Importance'] = importance_df['Importance'].round(4)
    
    st.dataframe(importance_df, use_container_width=True)
    
    # Feature analysis insights
    show_feature_insights(importance_df)

def show_feature_insights(importance_df: pd.DataFrame):
    """Show insights about feature importance"""
    
    st.markdown("### 💡 Feature Insights")
    
    # Top features
    top_features = importance_df.head(3)['Feature'].tolist()
    st.success(f"🎯 **Top 3 features**: {', '.join(top_features)}")
    
    # Feature distribution analysis
    total_importance = importance_df['Importance'].sum()
    top_5_importance = importance_df.head(5)['Importance'].sum()
    top_5_percentage = (top_5_importance / total_importance) * 100
    
    st.info(f"📊 **Top 5 features** account for **{top_5_percentage:.1f}%** of total importance")
    
    # Feature concentration
    if top_5_percentage > 80:
        st.warning("⚠️ High feature concentration - model relies heavily on few features")
    elif top_5_percentage < 40:
        st.info("📈 Well-distributed feature importance - model uses many features")
    else:
        st.success("✅ Balanced feature importance distribution")
    
    # Low importance features
    low_importance_features = importance_df[importance_df['Importance'] < 0.01]['Feature'].tolist()
    
    if low_importance_features:
        with st.expander(f"🗑️ Low Importance Features ({len(low_importance_features)} features)"):
            st.write("Consider removing these features to simplify your model:")
            for feature in low_importance_features[:10]:  # Show max 10
                st.write(f"• {feature}")
            
            if len(low_importance_features) > 10:
                st.write(f"... and {len(low_importance_features) - 10} more")

def show_model_interpretation():
    """Display model interpretation and explainability"""
    
    st.markdown("## 🧠 Model Interpretation")
    st.markdown("Understanding how your model makes decisions")
    
    model_name = st.session_state.model_name
    model = st.session_state.model
    
    # Model-specific interpretations
    st.markdown(f"### 🎯 {model_name} Interpretation")
    
    if model_name == "Logistic Regression":
        show_logistic_regression_interpretation()
    elif model_name == "Decision Tree":
        show_decision_tree_interpretation()
    elif model_name in ["Random Forest"]:
        show_ensemble_interpretation()
    else:
        show_general_interpretation()
    
    # SHAP analysis (if available)
    if st.checkbox("🔍 Advanced SHAP Analysis (may take time)"):
        try:
            show_shap_analysis()
        except Exception as e:
            st.warning(f"SHAP analysis not available: {str(e)}")

def show_logistic_regression_interpretation():
    """Show interpretation for logistic regression"""
    
    model = st.session_state.model
    feature_names = st.session_state.feature_names
    
    try:
        # Get coefficients
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            # Create coefficient plot
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef,
                'Abs_Coefficient': np.abs(coef)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            fig = px.bar(
                coef_df.head(15),
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Logistic Regression Coefficients',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Coefficient Interpretation")
            st.info("""
            **Positive coefficients** increase the probability of the positive class.
            **Negative coefficients** decrease the probability of the positive class.
            **Larger absolute values** have stronger influence on predictions.
            """)
            
            # Show coefficient table
            st.dataframe(coef_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error interpreting logistic regression: {str(e)}")

def show_decision_tree_interpretation():
    """Show interpretation for decision tree"""
    
    st.markdown("#### 🌳 Decision Tree Structure")
    
    model = st.session_state.model
    
    try:
        # Tree statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tree Depth", model.get_depth())
        with col2:
            st.metric("Number of Leaves", model.get_n_leaves())
        with col3:
            st.metric("Number of Nodes", model.tree_.node_count)
        
        st.info("""
        **Decision Trees** make predictions by following a series of if-else conditions.
        Each internal node represents a feature test, and each leaf represents a class prediction.
        The path from root to leaf explains the reasoning for each prediction.
        """)
        
        # Feature importance already shown in feature analysis
        st.markdown("Feature importance shows which features are most useful for splitting the data.")
        
    except Exception as e:
        st.error(f"Error interpreting decision tree: {str(e)}")

def show_ensemble_interpretation():
    """Show interpretation for ensemble models"""
    
    st.markdown("#### 🌲 Ensemble Model Interpretation")
    
    model = st.session_state.model
    model_name = st.session_state.model_name
    
    try:
        if model_name == "Random Forest":
            st.metric("Number of Trees", model.n_estimators)
            
            st.info("""
            **Random Forest** combines predictions from multiple decision trees.
            Each tree is trained on a random subset of data and features.
            Final prediction is made by majority voting (classification) or averaging (regression).
            Feature importance represents the average importance across all trees.
            """)
        
        # Out-of-bag score for Random Forest
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            st.metric("Out-of-Bag Score", f"{model.oob_score_:.3f}")
            st.info("Out-of-bag score provides an unbiased estimate of model performance without separate validation.")
    
    except Exception as e:
        st.error(f"Error interpreting ensemble model: {str(e)}")

def show_general_interpretation():
    """Show general interpretation for other models"""
    
    model_name = st.session_state.model_name
    
    interpretations = {
        "Support Vector Machine": """
        **SVM** finds the optimal boundary (hyperplane) that separates different classes.
        It maximizes the margin between classes and can use different kernels for non-linear separation.
        Feature importance is not directly available, but coefficient analysis can provide insights for linear kernels.
        """,
        
        "K-Nearest Neighbors": """
        **KNN** makes predictions based on the k nearest neighbors in the feature space.
        It's a lazy learning algorithm that doesn't build an explicit model.
        Predictions are made by majority voting among neighbors.
        Feature scaling is crucial for good performance.
        """,
        
        "Naive Bayes": """
        **Naive Bayes** assumes features are independent and uses Bayes' theorem for classification.
        It calculates the probability of each class given the features.
        Despite the strong independence assumption, it often works well in practice.
        Feature importance reflects how much each feature contributes to class separation.
        """
    }
    
    if model_name in interpretations:
        st.info(interpretations[model_name])
    else:
        st.info("Model interpretation varies by algorithm type. Check the documentation for specific details.")

def show_shap_analysis():
    """Show SHAP analysis for model interpretation"""
    
    st.markdown("#### 🔍 SHAP (SHapley Additive exPlanations) Analysis")
    
    try:
        import shap
        
        model = st.session_state.model
        X_test = st.session_state.X_test
        
        # Create explainer
        if st.session_state.model_name in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test)
        
        # Calculate SHAP values
        with st.spinner("Calculating SHAP values..."):
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Limit for performance
        
        st.success("✅ SHAP analysis completed!")
        
        # Summary plot would go here (requires matplotlib integration)
        st.info("SHAP analysis provides detailed feature contribution for each prediction.")
        
    except ImportError:
        st.warning("SHAP library not available. Install with: pip install shap")
    except Exception as e:
        st.error(f"SHAP analysis error: {str(e)}")

def show_detailed_report():
    """Show comprehensive model evaluation report"""
    
    st.markdown("## 📋 Detailed Evaluation Report")
    st.markdown("Comprehensive summary of your model's performance")
    
    model_name = st.session_state.model_name
    metrics = st.session_state.metrics
    
    # Report header
    st.markdown(f"### 📊 Model Evaluation Report: {model_name}")
    st.markdown(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Executive summary
    st.markdown("### 🎯 Executive Summary")
    
    accuracy = metrics.get('Accuracy', 0)
    if accuracy >= 0.9:
        summary = "Excellent performance with high accuracy across all metrics."
    elif accuracy >= 0.8:
        summary = "Good performance with solid accuracy and reliable predictions."
    elif accuracy >= 0.7:
        summary = "Fair performance that may benefit from additional optimization."
    else:
        summary = "Performance needs improvement through feature engineering or algorithm selection."
    
    st.info(f"**Overall Assessment:** {summary}")
    
    # Detailed sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Performance Metrics")
        for metric, value in metrics.items():
            st.write(f"• **{metric}**: {value:.3f}")
    
    with col2:
        st.markdown("#### ⚙️ Model Configuration")
        st.write(f"• **Algorithm**: {model_name}")
        st.write(f"• **Training Samples**: {len(st.session_state.X_train)}")
        st.write(f"• **Test Samples**: {len(st.session_state.X_test)}")
        st.write(f"• **Features**: {len(st.session_state.feature_names)}")
        st.write(f"• **Training Time**: {st.session_state.get('training_time', 0):.2f}s")
    
    # Classification report
    if st.session_state.predictions is not None:
        st.markdown("### 📊 Detailed Classification Report")
        
        report = classification_report(
            st.session_state.y_test,
            st.session_state.predictions,
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = generate_recommendations(model_name, metrics)
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Export report
    if st.button("📥 Download Report"):
        report_content = generate_report_content(model_name, metrics)
        st.download_button(
            label="Download Detailed Report",
            data=report_content,
            file_name=f"model_evaluation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

def generate_recommendations(model_name: str, metrics: dict) -> list:
    """Generate recommendations based on model performance"""
    
    recommendations = []
    accuracy = metrics.get('Accuracy', 0)
    precision = metrics.get('Precision', 0)
    recall = metrics.get('Recall', 0)
    
    # General recommendations
    if accuracy < 0.8:
        recommendations.append("Consider feature engineering or trying different algorithms to improve performance.")
    
    if precision < recall - 0.1:
        recommendations.append("Low precision indicates many false positives. Consider adjusting the decision threshold.")
    elif recall < precision - 0.1:
        recommendations.append("Low recall indicates missed positive cases. Consider class balancing techniques.")
    
    # Model-specific recommendations
    if model_name == "Logistic Regression" and accuracy < 0.8:
        recommendations.append("Try polynomial features or feature interactions for better linear separation.")
    
    if model_name == "Decision Tree" and accuracy > 0.95:
        recommendations.append("Very high accuracy might indicate overfitting. Consider cross-validation.")
    
    if model_name == "Random Forest":
        recommendations.append("Random Forest is performing well. Consider hyperparameter tuning for optimization.")
    
    if model_name == "K-Nearest Neighbors" and accuracy < 0.8:
        recommendations.append("Ensure features are properly scaled and consider different k values.")
    
    # Data recommendations
    if len(st.session_state.X_train) < 1000:
        recommendations.append("Consider collecting more training data for better model generalization.")
    
    recommendations.append("Validate model performance on additional datasets before deployment.")
    
    return recommendations

def generate_report_content(model_name: str, metrics: dict) -> str:
    """Generate downloadable report content"""
    
    report_lines = [
        f"Model Evaluation Report: {model_name}",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        "",
        "PERFORMANCE METRICS:",
    ]
    
    for metric, value in metrics.items():
        report_lines.append(f"  {metric}: {value:.3f}")
    
    report_lines.extend([
        "",
        "MODEL CONFIGURATION:",
        f"  Algorithm: {model_name}",
        f"  Training Samples: {len(st.session_state.X_train)}",
        f"  Test Samples: {len(st.session_state.X_test)}",
        f"  Features: {len(st.session_state.feature_names)}",
        f"  Training Time: {st.session_state.get('training_time', 0):.2f}s",
        "",
        "RECOMMENDATIONS:",
    ])
    
    recommendations = generate_recommendations(model_name, metrics)
    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"  {i}. {rec}")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    show()
