"""
Model Training Module
Comprehensive machine learning model training with hyperparameter tuning
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score, GridSearchCV
from utils.helpers import show_learning_tip, create_metrics_comparison_chart
from utils.model_utils import (
    get_available_models, create_hyperparameter_widgets, train_model,
    evaluate_classification_model, perform_cross_validation, get_feature_importance,
    save_model
)

def show():
    """Main function to display the model training interface"""
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please complete data preprocessing first!")
        return
    
    st.title("🧠 Model Training Module")
    st.markdown("Train and tune machine learning models on your preprocessed data!")
    
    # Learning tip
    show_learning_tip(
        "Start with simple models and gradually try more complex ones. "
        "Compare multiple algorithms to find the best performer for your specific dataset.",
        "info"
    )
    
    # Training sections
    training_tabs = st.tabs([
        "🎯 Model Selection",
        "🎛️ Hyperparameter Tuning", 
        "🚀 Model Training",
        "📊 Model Comparison",
        "💾 Model Management"
    ])
    
    with training_tabs[0]:
        show_model_selection()
    
    with training_tabs[1]:
        show_hyperparameter_tuning()
    
    with training_tabs[2]:
        show_model_training()
    
    with training_tabs[3]:
        show_model_comparison()
    
    with training_tabs[4]:
        show_model_management()

def show_model_selection():
    """Display model selection interface"""
    
    st.markdown("## 🎯 Model Selection")
    st.markdown("Choose the best algorithm for your machine learning task")
    
    # Get available models
    models = get_available_models()
    
    # Display model information
    st.markdown("### 📚 Available Algorithms")
    
    # Create tabs for each model
    model_tabs = st.tabs(list(models.keys()))
    
    for i, (model_name, model_info) in enumerate(models.items()):
        with model_tabs[i]:
            display_model_info(model_name, model_info)
    
    # Model selection
    st.markdown("### 🎯 Select Model to Train")
    
    selected_model = st.selectbox(
        "Choose an algorithm:",
        list(models.keys()),
        help="Select the machine learning algorithm you want to train"
    )
    
    if selected_model:
        st.session_state.selected_model = selected_model
        st.success(f"✅ Selected: **{selected_model}**")
        
        # Show model-specific information
        model_info = models[selected_model]
        
        with st.expander(f"📖 About {selected_model}"):
            st.write(model_info["description"])
            
            # When to use this model
            usage_tips = get_model_usage_tips(selected_model)
            st.markdown("**When to use:**")
            for tip in usage_tips:
                st.write(f"• {tip}")
            
            # Pros and cons
            pros_cons = get_model_pros_cons(selected_model)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pros:**")
                for pro in pros_cons["pros"]:
                    st.write(f"✅ {pro}")
            
            with col2:
                st.markdown("**Cons:**")
                for con in pros_cons["cons"]:
                    st.write(f"❌ {con}")

def display_model_info(model_name: str, model_info: dict):
    """Display detailed information about a model"""
    
    st.markdown(f"### {model_name}")
    st.write(model_info["description"])
    
    # Algorithm complexity
    complexity_info = get_algorithm_complexity(model_name)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Speed", complexity_info["training_speed"])
    with col2:
        st.metric("Prediction Speed", complexity_info["prediction_speed"])
    with col3:
        st.metric("Interpretability", complexity_info["interpretability"])
    
    # Sample hyperparameters
    if model_info.get("hyperparams"):
        st.markdown("**Key Hyperparameters:**")
        for param, config in list(model_info["hyperparams"].items())[:3]:
            st.write(f"• **{param}**: {config.get('default', 'N/A')}")

def get_algorithm_complexity(model_name: str) -> dict:
    """Get algorithm complexity information"""
    
    complexity_map = {
        "Logistic Regression": {"training_speed": "Fast", "prediction_speed": "Fast", "interpretability": "High"},
        "Decision Tree": {"training_speed": "Fast", "prediction_speed": "Fast", "interpretability": "High"},
        "Random Forest": {"training_speed": "Medium", "prediction_speed": "Medium", "interpretability": "Medium"},
        "Support Vector Machine": {"training_speed": "Slow", "prediction_speed": "Fast", "interpretability": "Low"},
        "K-Nearest Neighbors": {"training_speed": "Fast", "prediction_speed": "Slow", "interpretability": "Medium"},
        "Naive Bayes": {"training_speed": "Fast", "prediction_speed": "Fast", "interpretability": "Medium"}
    }
    
    return complexity_map.get(model_name, {"training_speed": "Unknown", "prediction_speed": "Unknown", "interpretability": "Unknown"})

def get_model_usage_tips(model_name: str) -> list:
    """Get usage tips for each model"""
    
    tips_map = {
        "Logistic Regression": [
            "Linear relationships between features and target",
            "Binary or multi-class classification",
            "When you need interpretable results",
            "Small to medium datasets"
        ],
        "Decision Tree": [
            "Non-linear relationships",
            "Mixed data types (numerical and categorical)",
            "When interpretability is important",
            "Feature interactions matter"
        ],
        "Random Forest": [
            "Good general-purpose algorithm",
            "Handles missing values well",
            "Reduces overfitting compared to single trees",
            "When you have many features"
        ],
        "Support Vector Machine": [
            "High-dimensional data",
            "Clear margin of separation exists",
            "Memory efficient",
            "Versatile with different kernels"
        ],
        "K-Nearest Neighbors": [
            "Simple, intuitive algorithm",
            "Non-parametric problems",
            "Local patterns in data",
            "When training data is representative"
        ],
        "Naive Bayes": [
            "Text classification",
            "Small datasets",
            "Fast training and prediction",
            "Features are relatively independent"
        ]
    }
    
    return tips_map.get(model_name, ["General purpose algorithm"])

def get_model_pros_cons(model_name: str) -> dict:
    """Get pros and cons for each model"""
    
    pros_cons_map = {
        "Logistic Regression": {
            "pros": ["Simple and interpretable", "Fast training", "No hyperparameter tuning needed", "Probabilistic output"],
            "cons": ["Assumes linear relationship", "Sensitive to outliers", "Requires feature scaling", "Limited complexity"]
        },
        "Decision Tree": {
            "pros": ["Highly interpretable", "Handles non-linear data", "No scaling needed", "Built-in feature selection"],
            "cons": ["Prone to overfitting", "Unstable", "Biased toward features with many levels", "Poor extrapolation"]
        },
        "Random Forest": {
            "pros": ["Reduces overfitting", "Handles missing values", "Feature importance", "Good performance"],
            "cons": ["Less interpretable", "Can overfit with noisy data", "Memory intensive", "Not good for linear relationships"]
        },
        "Support Vector Machine": {
            "pros": ["Effective in high dimensions", "Memory efficient", "Versatile kernels", "Good generalization"],
            "cons": ["Slow on large datasets", "Sensitive to feature scaling", "No probabilistic output", "Hard to interpret"]
        },
        "K-Nearest Neighbors": {
            "pros": ["Simple algorithm", "No assumptions about data", "Good for irregular decision boundaries", "Handles multi-class naturally"],
            "cons": ["Computationally expensive", "Sensitive to irrelevant features", "Requires feature scaling", "Poor with high dimensions"]
        },
        "Naive Bayes": {
            "pros": ["Very fast", "Good with small datasets", "Handles multi-class well", "Good baseline"],
            "cons": ["Strong independence assumption", "Poor estimator for probability", "Categorical inputs need smoothing", "Limited by feature independence"]
        }
    }
    
    return pros_cons_map.get(model_name, {"pros": [], "cons": []})

def show_hyperparameter_tuning():
    """Display hyperparameter tuning interface"""
    
    st.markdown("## 🎛️ Hyperparameter Tuning")
    st.markdown("Fine-tune your model's performance by adjusting hyperparameters")
    
    if not hasattr(st.session_state, 'selected_model'):
        st.warning("⚠️ Please select a model first in the 'Model Selection' tab!")
        return
    
    selected_model = st.session_state.selected_model
    models = get_available_models()
    model_info = models[selected_model]
    
    st.info(f"🎯 Tuning hyperparameters for: **{selected_model}**")
    
    # Hyperparameter tuning method
    tuning_method = st.selectbox(
        "Select tuning method:",
        ["Manual Tuning", "Grid Search", "Random Search"],
        help="Choose how to optimize hyperparameters"
    )
    
    if tuning_method == "Manual Tuning":
        show_manual_tuning(selected_model, model_info)
    elif tuning_method == "Grid Search":
        show_grid_search_tuning(selected_model, model_info)
    else:
        show_random_search_tuning(selected_model, model_info)

def show_manual_tuning(model_name: str, model_info: dict):
    """Show manual hyperparameter tuning interface"""
    
    st.markdown("### 🎛️ Manual Hyperparameter Tuning")
    st.markdown("Adjust hyperparameters manually using sliders and dropdowns")
    
    # Create hyperparameter widgets
    params = create_hyperparameter_widgets(model_name, model_info)
    
    # Store parameters
    st.session_state.model_params = params
    
    # Show parameter summary
    with st.expander("📋 Current Parameter Settings"):
        for param, value in params.items():
            st.write(f"**{param}**: {value}")
    
    # Quick cross-validation
    if st.button("🔄 Quick Cross-Validation"):
        with st.spinner("Running cross-validation..."):
            cv_results = perform_cross_validation(
                model_info["class"], 
                params, 
                st.session_state.X_train, 
                st.session_state.y_train,
                cv=5
            )
            
            if cv_results:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CV Mean Accuracy", f"{cv_results['CV Mean Accuracy']:.3f}")
                with col2:
                    st.metric("CV Std Accuracy", f"{cv_results['CV Std Accuracy']:.3f}")
                
                # Plot CV scores
                fig = px.bar(
                    x=range(1, 6),
                    y=cv_results['CV Scores'],
                    title="Cross-Validation Scores",
                    labels={'x': 'Fold', 'y': 'Accuracy'}
                )
                st.plotly_chart(fig, use_container_width=True)

def show_grid_search_tuning(model_name: str, model_info: dict):
    """Show grid search hyperparameter tuning"""
    
    st.markdown("### 🔍 Grid Search Tuning")
    st.markdown("Systematically search through hyperparameter combinations")
    
    st.warning("⚠️ Grid search can be computationally expensive for large parameter spaces!")
    
    # Define parameter grids
    param_grids = get_parameter_grids(model_name)
    
    st.markdown("#### 📋 Parameter Grid")
    for param, values in param_grids.items():
        st.write(f"**{param}**: {values}")
    
    # Grid search options
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
    with col2:
        scoring = st.selectbox("Scoring metric:", ["accuracy", "f1_weighted", "roc_auc"])
    
    if st.button("🚀 Run Grid Search"):
        with st.spinner("Running grid search... This may take a while."):
            try:
                grid_search = GridSearchCV(
                    model_info["class"](),
                    param_grids,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                grid_search.fit(st.session_state.X_train, st.session_state.y_train)
                
                st.success("✅ Grid search completed!")
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Best CV Score", f"{grid_search.best_score_:.3f}")
                with col2:
                    st.metric("Best Parameters", "See below")
                
                st.markdown("#### 🎯 Best Parameters")
                for param, value in grid_search.best_params_.items():
                    st.write(f"**{param}**: {value}")
                
                # Store best parameters
                st.session_state.model_params = grid_search.best_params_
                
                # Show top results
                results_df = pd.DataFrame(grid_search.cv_results_)
                top_results = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
                
                st.markdown("#### 📊 Top 5 Results")
                st.dataframe(top_results, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in grid search: {str(e)}")

def show_random_search_tuning(model_name: str, model_info: dict):
    """Show random search hyperparameter tuning"""
    
    st.markdown("### 🎲 Random Search Tuning")
    st.markdown("Randomly sample hyperparameter combinations")
    
    st.info("💡 Random search is often more efficient than grid search for high-dimensional spaces!")
    
    # Random search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_iter = st.slider("Number of iterations:", 10, 100, 20)
    with col2:
        cv_folds = st.slider("CV folds:", 3, 10, 5)
    with col3:
        scoring = st.selectbox("Scoring:", ["accuracy", "f1_weighted", "roc_auc"])
    
    st.info(f"Will test {n_iter} random parameter combinations with {cv_folds}-fold cross-validation")
    
    if st.button("🎲 Run Random Search"):
        st.info("Random search functionality would be implemented here with RandomizedSearchCV")

def get_parameter_grids(model_name: str) -> dict:
    """Get parameter grids for grid search"""
    
    param_grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        },
        "Support Vector Machine": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"]
        },
        "Naive Bayes": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    }
    
    return param_grids.get(model_name, {})

def show_model_training():
    """Display model training interface"""
    
    st.markdown("## 🚀 Model Training")
    st.markdown("Train your selected model with chosen hyperparameters")
    
    if not hasattr(st.session_state, 'selected_model'):
        st.warning("⚠️ Please select a model first!")
        return
    
    selected_model = st.session_state.selected_model
    models = get_available_models()
    model_info = models[selected_model]
    
    # Validate feature matrix and target
    if st.session_state.target_name in st.session_state.X_train.columns:
        # Ensure target is not among features
        st.session_state.X_train = st.session_state.X_train.drop(columns=[st.session_state.target_name], errors='ignore')
        st.session_state.X_test = st.session_state.X_test.drop(columns=[st.session_state.target_name], errors='ignore')
        st.session_state.feature_names = st.session_state.X_train.columns.tolist()

    # Ensure all features are numeric
    non_numeric_cols = st.session_state.X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.error(
            "Non-numeric features detected in X. Please finish categorical encoding in Preprocessing. "
            f"Columns: {', '.join(non_numeric_cols)}"
        )
        return

    # Get parameters
    params = st.session_state.get('model_params', model_info['params'])
    
    st.info(f"🎯 Training: **{selected_model}**")
    
    # Show training configuration
    with st.expander("📋 Training Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Training samples: {len(st.session_state.X_train)}")
            st.write(f"- Features: {len(st.session_state.feature_names)}")
            st.write(f"- Target: {st.session_state.target_name}")
        
        with col2:
            st.write("**Model Parameters:**")
            for param, value in params.items():
                st.write(f"- {param}: {value}")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        perform_cv = st.checkbox("Perform cross-validation", value=True)
    with col2:
        if perform_cv:
            cv_folds = st.slider("CV folds:", 3, 10, 5)
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        train_model_with_progress(selected_model, model_info, params, perform_cv, cv_folds if perform_cv else 5)

def train_model_with_progress(model_name: str, model_info: dict, params: dict, perform_cv: bool, cv_folds: int):
    """Train model with progress tracking"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize model
        status_text.text("🔧 Initializing model...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Step 2: Cross-validation (if requested)
        if perform_cv:
            status_text.text("🔄 Performing cross-validation...")
            progress_bar.progress(40)
            
            cv_results = perform_cross_validation(
                model_info["class"], 
                params, 
                st.session_state.X_train, 
                st.session_state.y_train,
                cv=cv_folds
            )
            
            if cv_results:
                st.success(f"✅ Cross-validation completed! Mean accuracy: {cv_results['CV Mean Accuracy']:.3f} ± {cv_results['CV Std Accuracy']:.3f}")
        
        # Step 3: Train final model
        status_text.text("🚀 Training final model...")
        progress_bar.progress(70)
        
        start_time = time.time()
        model = train_model(
            model_info["class"], 
            params, 
            st.session_state.X_train, 
            st.session_state.y_train
        )
        training_time = time.time() - start_time
        
        if model is None:
            st.error("❌ Model training failed!")
            return
        
        # Step 4: Evaluate model
        status_text.text("📊 Evaluating model...")
        progress_bar.progress(90)
        
        metrics, y_pred, y_pred_proba = evaluate_classification_model(
            model, 
            st.session_state.X_test, 
            st.session_state.y_test
        )
        
        # Step 5: Complete
        status_text.text("✅ Training completed!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.model = model
        st.session_state.model_name = model_name
        st.session_state.model_params = params
        st.session_state.training_time = training_time
        st.session_state.metrics = metrics
        st.session_state.predictions = y_pred
        st.session_state.prediction_probabilities = y_pred_proba
        
        # Display results
        display_training_results(model_name, metrics, training_time, cv_results if perform_cv else None)
        
        # Feature importance
        feature_importance = get_feature_importance(model, st.session_state.feature_names)
        if feature_importance is not None:
            st.session_state.feature_importance = feature_importance
            display_feature_importance(feature_importance)
        
    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_training_results(model_name: str, metrics: dict, training_time: float, cv_results: dict = None):
    """Display training results"""
    
    st.markdown("---")
    st.markdown("## 🎉 Training Results")
    
    # Training summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", model_name)
    with col2:
        st.metric("Training Time", f"{training_time:.2f}s")
    with col3:
        st.metric("Test Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics.get('F1-Score', 0):.3f}")
    
    # Detailed metrics
    st.markdown("### 📊 Detailed Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("**Classification Metrics:**")
        for metric, value in metrics.items():
            st.write(f"• **{metric}**: {value:.3f}")
    
    with metrics_col2:
        # Metrics radar chart
        if len(metrics) > 3:
            fig = create_metrics_comparison_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation results
    if cv_results:
        st.markdown("### 🔄 Cross-Validation Results")
        
        cv_col1, cv_col2 = st.columns(2)
        
        with cv_col1:
            st.metric("CV Mean Accuracy", f"{cv_results['CV Mean Accuracy']:.3f}")
            st.metric("CV Std Accuracy", f"{cv_results['CV Std Accuracy']:.3f}")
        
        with cv_col2:
            fig = px.bar(
                x=range(1, len(cv_results['CV Scores']) + 1),
                y=cv_results['CV Scores'],
                title="Cross-Validation Fold Scores",
                labels={'x': 'Fold', 'y': 'Accuracy'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(feature_importance: np.ndarray):
    """Display feature importance"""
    
    st.markdown("### 🎯 Feature Importance")
    
    feature_names = st.session_state.feature_names
    
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Display top features
    top_n = min(15, len(feature_names))
    
    importance_data = {
        'Feature': [feature_names[i] for i in indices[:top_n]],
        'Importance': feature_importance[indices[:top_n]]
    }
    
    fig = px.bar(
        importance_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n} Feature Importances"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    importance_df = pd.DataFrame(importance_data)
    st.dataframe(importance_df, use_container_width=True)

def show_model_comparison():
    """Show model comparison interface"""
    
    st.markdown("## 📊 Model Comparison")
    st.markdown("Compare different models and their performance")
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = []
    
    # Add current model to comparison
    if st.session_state.get('model') is not None:
        if st.button("➕ Add Current Model to Comparison"):
            model_info = {
                'name': st.session_state.model_name,
                'model': st.session_state.model,
                'metrics': st.session_state.metrics,
                'training_time': st.session_state.training_time,
                'params': st.session_state.model_params
            }
            st.session_state.trained_models.append(model_info)
            st.success(f"✅ Added {st.session_state.model_name} to comparison!")
    
    # Display comparison
    if st.session_state.trained_models:
        display_model_comparison_table()
        display_model_comparison_charts()
    else:
        st.info("🔄 Train some models first to enable comparison!")

def display_model_comparison_table():
    """Display model comparison table"""
    
    st.markdown("### 📋 Model Comparison Table")
    
    comparison_data = []
    for model_info in st.session_state.trained_models:
        row = {
            'Model': model_info['name'],
            'Training Time (s)': f"{model_info['training_time']:.2f}",
        }
        row.update({metric: f"{value:.3f}" for metric, value in model_info['metrics'].items()})
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Highlight best models
    if len(comparison_data) > 1:
        st.markdown("### 🏆 Best Performers")
        
        metrics_to_compare = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for metric in metrics_to_compare:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].astype(float).idxmax()
                best_model = comparison_df.loc[best_idx, 'Model']
                best_score = comparison_df.loc[best_idx, metric]
                st.write(f"🥇 **Best {metric}**: {best_model} ({best_score})")

def display_model_comparison_charts():
    """Display model comparison charts"""
    
    st.markdown("### 📊 Performance Comparison Charts")
    
    if len(st.session_state.trained_models) < 2:
        st.info("Add more models to see comparison charts!")
        return
    
    # Prepare data for plotting
    model_names = [model['name'] for model in st.session_state.trained_models]
    
    # Accuracy comparison
    accuracies = [model['metrics'].get('Accuracy', 0) for model in st.session_state.trained_models]
    
    fig1 = px.bar(
        x=model_names,
        y=accuracies,
        title="Model Accuracy Comparison",
        labels={'x': 'Model', 'y': 'Accuracy'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Training time comparison
    training_times = [model['training_time'] for model in st.session_state.trained_models]
    
    fig2 = px.bar(
        x=model_names,
        y=training_times,
        title="Training Time Comparison",
        labels={'x': 'Model', 'y': 'Training Time (seconds)'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Multi-metric comparison
    metrics_data = []
    for model in st.session_state.trained_models:
        for metric, value in model['metrics'].items():
            metrics_data.append({
                'Model': model['name'],
                'Metric': metric,
                'Value': value
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        fig3 = px.bar(
            metrics_df,
            x='Model',
            y='Value',
            color='Metric',
            title="All Metrics Comparison",
            barmode='group'
        )
        st.plotly_chart(fig3, use_container_width=True)

def show_model_management():
    """Show model management interface"""
    
    st.markdown("## 💾 Model Management")
    st.markdown("Save, load, and manage your trained models")
    
    # Save current model
    if st.session_state.get('model') is not None:
        st.markdown("### 💾 Save Current Model")
        
        model_name_input = st.text_input(
            "Model name:",
            value=f"{st.session_state.model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
        )
        
        if st.button("💾 Save Model"):
            if model_name_input:
                filepath = save_model(
                    st.session_state.model,
                    model_name_input,
                    st.session_state.feature_names,
                    st.session_state.target_name
                )
                
                if filepath:
                    st.success(f"✅ Model saved to: {filepath}")
                    
                    # Model download
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="📥 Download Model",
                            data=f.read(),
                            file_name=f"{model_name_input}.joblib",
                            mime="application/octet-stream"
                        )
            else:
                st.warning("⚠️ Please enter a model name!")
    else:
        st.info("🔄 Train a model first to enable saving!")
    
    # Model export summary
    if st.session_state.get('model') is not None:
        st.markdown("### 📋 Model Export Summary")
        
        export_info = {
            "Model Type": st.session_state.model_name,
            "Features": len(st.session_state.feature_names),
            "Target": st.session_state.target_name,
            "Training Accuracy": f"{st.session_state.metrics.get('Accuracy', 0):.3f}",
            "Export Date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for key, value in export_info.items():
            st.write(f"**{key}**: {value}")

if __name__ == "__main__":
    show()
