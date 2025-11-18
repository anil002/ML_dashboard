"""
🤖 Interactive Machine Learning Dashboard
An educational and comprehensive ML platform built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import custom modules
from modules import (
    data_upload, eda, preprocessing, model_training, evaluation, 
    prediction, explanations, advanced_data_upload, interactive_animations, 
    user_experience, algorithm_guide
)
from utils import helpers

# Page configuration
st.set_page_config(
    page_title="ML Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .module-header {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Interactive Machine Learning Dashboard</h1>
        <p>Learn, Experiment, and Master Machine Learning Concepts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'target_name' not in st.session_state:
        st.session_state.target_name = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Module selection
    modules = {
        "🏠 Home": "home",
        "📁 Data Upload": "data_upload",
        "� Advanced Data Upload": "advanced_data_upload",
        "�🔍 Exploratory Data Analysis": "eda",
        "⚙️ Data Preprocessing": "preprocessing",
        "🧠 Model Training": "model_training",
        "📈 Model Evaluation": "evaluation",
        "🔮 Predictions": "prediction",
        "🎬 Interactive Animations": "interactive_animations",
        "� Personalized Experience": "user_experience",
        "�📚 Algorithm Guide": "algorithm_guide"
    }
    
    selected_module = st.sidebar.selectbox(
        "Select Module",
        list(modules.keys()),
        index=0
    )
    
    current_module = modules[selected_module]
    
    # Progress tracker
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Progress Tracker")
    
    progress_items = {
        "Data Loaded": st.session_state.data is not None,
        "Data Preprocessed": st.session_state.processed_data is not None,
        "Model Trained": st.session_state.model is not None,
        "Predictions Made": st.session_state.predictions is not None
    }
    
    for item, status in progress_items.items():
        if status:
            st.sidebar.success(f"✅ {item}")
        else:
            st.sidebar.info(f"⏳ {item}")
    
    # Module routing
    if current_module == "home":
        show_home()
    elif current_module == "data_upload":
        data_upload.show()
    elif current_module == "advanced_data_upload":
        advanced_data_upload.show()
    elif current_module == "eda":
        eda.show()
    elif current_module == "preprocessing":
        preprocessing.show()
    elif current_module == "model_training":
        model_training.show()
    elif current_module == "evaluation":
        evaluation.show()
    elif current_module == "prediction":
        prediction.show()
    elif current_module == "interactive_animations":
        interactive_animations.show()
    elif current_module == "user_experience":
        user_experience.show()
    elif current_module == "algorithm_guide":
        algorithm_guide.show()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 **Tip**: Hover over any element for helpful tooltips!"
    )
    st.sidebar.markdown(
        "🎓 **Learning Mode**: Each section includes educational content to help you understand ML concepts."
    )

def show_home():
    """Display the home page"""
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Your ML Learning Journey! 🚀
        
        This interactive dashboard is designed to help you learn and master machine learning concepts 
        through hands-on experimentation. Whether you're a beginner or an advanced practitioner, 
        you'll find valuable tools and insights here.
        
        ### 🎯 What You Can Do:
        - **Upload and explore** your own datasets
        - **Preprocess data** with various techniques
        - **Train multiple ML models** with different algorithms
        - **Evaluate and compare** model performance
        - **Make predictions** on new data
        - **Learn concepts** through interactive explanations
        """)
    
    with col2:
        st.info("""
        ### 🎓 Learning Features
        - Step-by-step guidance
        - Algorithm explanations
        - Interactive visualizations
        - Best practices tips
        - Real-time feedback
        """)
    
    # Quick start guide
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Guide")
    
    steps = [
        ("1️⃣", "Data Upload", "Upload your CSV or Excel file to get started"),
        ("2️⃣", "Explore Data", "Understand your data through visualizations and statistics"),
        ("3️⃣", "Preprocess", "Clean and prepare your data for modeling"),
        ("4️⃣", "Train Model", "Select and train machine learning algorithms"),
        ("5️⃣", "Evaluate", "Assess model performance with various metrics"),
        ("6️⃣", "Predict", "Make predictions on new data")
    ]
    
    cols = st.columns(3)
    for i, (emoji, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{emoji} {title}</h4>
                <p style="font-size: 14px; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample datasets
    st.markdown("---")
    st.markdown("## 📊 Try with Sample Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌸 Load Iris Dataset (Classification)"):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            st.session_state.data = df
            st.session_state.data_source = "Sample: Iris Dataset"
            st.success("✅ Iris dataset loaded! Navigate to 'Exploratory Data Analysis' to explore.")
    
    with col2:
        if st.button("🏠 Load Boston Housing Dataset (Regression)"):
            try:
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['target'] = boston.target
                st.session_state.data = df
                st.session_state.data_source = "Sample: Boston Housing Dataset"
                st.success("✅ Boston Housing dataset loaded! Navigate to 'Exploratory Data Analysis' to explore.")
            except ImportError:
                st.warning("Boston Housing dataset is not available in newer versions of scikit-learn. Please upload your own dataset.")
    
    with col3:
        if st.button("🚀 Try Advanced Features"):
            st.info("🎉 Check out the '🚀 Advanced Data Upload' for multiple data sources, or '🎬 Interactive Animations' to see ML in action!")
    
    # Quick navigation hints
    st.markdown("---")
    st.markdown("## 🧭 Quick Navigation Tips")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        ### 🎯 **For Beginners:**
        1. Start with **📁 Data Upload** to load your data
        2. Explore with **🔍 Exploratory Data Analysis**
        3. Clean data in **⚙️ Data Preprocessing**
        4. Train models in **🧠 Model Training**
        5. Check out **📚 Algorithm Guide** for learning
        """)
    
    with nav_col2:
        st.markdown("""
        ### 🚀 **For Advanced Users:**
        1. Try **🚀 Advanced Data Upload** for multiple sources
        2. Watch **🎬 Interactive Animations** of ML processes
        3. Customize with **👤 Personalized Experience**
        4. Use **📈 Model Evaluation** for deep analysis
        5. Make **🔮 Predictions** on new data
        """)
    
    # Welcome message for new users
    if 'data' not in st.session_state or st.session_state.data is None:
        st.info("💡 **New here?** Try loading one of the sample datasets above, or upload your own data using the sidebar navigation!")
    
    # Learning resources
    st.markdown("---")
    st.markdown("## 📚 Learning Resources")
    
    resources_col1, resources_col2 = st.columns(2)
    
    with resources_col1:
        st.markdown("""
        ### 🎓 Concepts You'll Learn:
        - Data preprocessing techniques
        - Feature engineering
        - Model selection and validation
        - Hyperparameter tuning
        - Performance evaluation
        - Model interpretation
        """)
    
    with resources_col2:
        st.markdown("""
        ### 🛠️ Algorithms Available:
        - **Logistic Regression**: Linear classification
        - **Decision Trees**: Rule-based decisions
        - **Random Forest**: Ensemble of trees
        - **SVM**: Support Vector Machines
        - **KNN**: k-Nearest Neighbors
        - **Naive Bayes**: Probabilistic classifier
        """)
    
    # Tips section
    st.markdown("---")
    st.info("""
    💡 **Pro Tips:**
    - Start with the sample datasets to familiarize yourself with the interface
    - Read the algorithm explanations in the 'Algorithm Guide' section
    - Experiment with different preprocessing techniques to see their impact
    - Compare multiple models to find the best performer for your data
    - Use the model evaluation tools to understand your model's strengths and weaknesses
    """)

if __name__ == "__main__":
    main()
