"""
Utility functions for the ML Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

def display_dataframe_info(df: pd.DataFrame) -> None:
    """Display comprehensive information about a DataFrame"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Data types
    st.subheader("📊 Column Information")
    
    # Create data types summary
    dtype_summary = []
    for col in df.columns:
        dtype_summary.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": df[col].count(),
            "Null Count": df[col].isnull().sum(),
            "Unique Values": df[col].nunique(),
            "Sample Values": str(df[col].dropna().iloc[:3].tolist()) if len(df[col].dropna()) > 0 else "No data"
        })
    
    dtype_df = pd.DataFrame(dtype_summary)
    st.dataframe(dtype_df, use_container_width=True)

def create_distribution_plot(df: pd.DataFrame, column: str, plot_type: str = "histogram") -> go.Figure:
    """Create distribution plots for numerical columns"""
    
    if plot_type == "histogram":
        fig = px.histogram(
            df, 
            x=column, 
            title=f"Distribution of {column}",
            marginal="box",
            nbins=30
        )
    elif plot_type == "box":
        fig = px.box(
            df, 
            y=column, 
            title=f"Box Plot of {column}"
        )
    elif plot_type == "violin":
        fig = px.violin(
            df, 
            y=column, 
            title=f"Violin Plot of {column}",
            box=True
        )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        title_x=0.5
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap"""
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )
    
    fig.update_layout(
        title_x=0.5,
        height=600
    )
    
    return fig

def create_pairplot(df: pd.DataFrame, target_col: Optional[str] = None, max_cols: int = 5) -> go.Figure:
    """Create a pairplot using plotly"""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit number of columns for performance
    if len(numerical_cols) > max_cols:
        numerical_cols = numerical_cols[:max_cols]
        st.warning(f"Showing only first {max_cols} numerical columns for performance. "
                  f"Original dataset has {len(df.select_dtypes(include=[np.number]).columns)} numerical columns.")
    
    if target_col and target_col in df.columns:
        fig = px.scatter_matrix(
            df[numerical_cols + [target_col]], 
            dimensions=numerical_cols,
            color=target_col,
            title="Pairplot with Target Variable"
        )
    else:
        fig = px.scatter_matrix(
            df[numerical_cols], 
            dimensions=numerical_cols,
            title="Pairplot of Numerical Variables"
        )
    
    fig.update_layout(
        height=600,
        title_x=0.5
    )
    
    return fig

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numerical columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """Calculate missing value percentage for each column"""
    missing_percent = (df.isnull().sum() / len(df)) * 100
    return missing_percent.sort_values(ascending=False)

def create_missing_value_plot(df: pd.DataFrame) -> go.Figure:
    """Create a visualization for missing values"""
    
    missing_percent = calculate_missing_percentage(df)
    missing_percent = missing_percent[missing_percent > 0]
    
    if len(missing_percent) == 0:
        # Create an empty plot with message
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found! 🎉",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=20, color="green")
        )
        fig.update_layout(
            title="Missing Values Analysis",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    fig = px.bar(
        x=missing_percent.values,
        y=missing_percent.index,
        orientation='h',
        title="Missing Values by Column (%)",
        labels={'x': 'Missing Percentage (%)', 'y': 'Columns'}
    )
    
    fig.update_layout(
        height=max(400, len(missing_percent) * 30),
        title_x=0.5
    )
    
    return fig

def format_number(num: float, decimal_places: int = 2) -> str:
    """Format numbers for display"""
    if num >= 1e6:
        return f"{num/1e6:.{decimal_places}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"

def create_feature_importance_plot(feature_names: List[str], importances: np.ndarray, max_features: int = 15) -> go.Figure:
    """Create feature importance plot"""
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:max_features]
    
    fig = px.bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        title=f"Top {min(max_features, len(feature_names))} Feature Importances",
        labels={'x': 'Importance', 'y': 'Features'}
    )
    
    fig.update_layout(
        height=max(400, len(indices) * 30),
        title_x=0.5,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def show_code_snippet(code: str, language: str = "python") -> None:
    """Display a code snippet with syntax highlighting"""
    st.code(code, language=language)

def create_metrics_comparison_chart(metrics_dict: Dict[str, float]) -> go.Figure:
    """Create a radar chart for model metrics comparison"""
    
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Model Performance Metrics",
        title_x=0.5,
        height=500
    )
    
    return fig

def download_button_with_data(data: Any, filename: str, mime_type: str, button_text: str) -> None:
    """Create a download button for data"""
    st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def create_confusion_matrix_plot(cm: np.ndarray, labels: List[str]) -> go.Figure:
    """Create an interactive confusion matrix heatmap"""
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        title_x=0.5,
        height=500
    )
    
    return fig

def safe_execute(func, default_value=None, error_message="An error occurred"):
    """Safely execute a function and handle errors gracefully"""
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return default_value

def validate_dataframe(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> Tuple[bool, str]:
    """Validate if DataFrame meets minimum requirements"""
    
    if df is None:
        return False, "No data provided"
    
    if len(df) < min_rows:
        return False, f"Dataset should have at least {min_rows} rows. Current: {len(df)}"
    
    if len(df.columns) < min_cols:
        return False, f"Dataset should have at least {min_cols} columns. Current: {len(df.columns)}"
    
    return True, "Dataset validation passed"

def show_learning_tip(tip: str, tip_type: str = "info") -> None:
    """Display a learning tip in a formatted box"""
    
    if tip_type == "info":
        st.info(f"💡 **Learning Tip**: {tip}")
    elif tip_type == "warning":
        st.warning(f"⚠️ **Important**: {tip}")
    elif tip_type == "success":
        st.success(f"✅ **Good Practice**: {tip}")
    else:
        st.info(f"📚 **Note**: {tip}")

def create_learning_progress_bar(current_step: int, total_steps: int, step_names: List[str]) -> None:
    """Create a visual progress bar for the learning journey"""
    
    progress = current_step / total_steps
    st.progress(progress)
    
    st.write(f"Step {current_step}/{total_steps}: {step_names[current_step-1] if current_step <= len(step_names) else 'Complete'}")
    
    # Show completed steps
    for i, step_name in enumerate(step_names[:current_step], 1):
        st.success(f"✅ Step {i}: {step_name}")
    
    # Show remaining steps
    for i, step_name in enumerate(step_names[current_step:], current_step + 1):
        st.info(f"⏳ Step {i}: {step_name}")

def get_sample_data_description(dataset_name: str) -> str:
    """Get description for sample datasets"""
    
    descriptions = {
        "iris": """
        **Iris Dataset**: A classic dataset for classification containing 150 samples of iris flowers 
        with 4 features (sepal length, sepal width, petal length, petal width) and 3 species classes.
        Perfect for learning classification algorithms!
        """,
        "boston": """
        **Boston Housing Dataset**: Contains information about housing in Boston with 13 features 
        and house prices as the target. Great for learning regression algorithms!
        """,
        "titanic": """
        **Titanic Dataset**: Passenger information from the Titanic with survival as the target variable.
        Excellent for learning data preprocessing and classification!
        """
    }
    
    return descriptions.get(dataset_name, "Sample dataset for machine learning practice.")
