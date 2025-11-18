"""
Interactive Process Visualization and Animation Module
Real-time animations for ML pipeline processes
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any
import json

def show():
    """Main function to display interactive animations interface"""
    
    st.title("🎬 Interactive ML Animations")
    st.markdown("Experience machine learning processes through engaging animations and visualizations")
    
    # Animation selection tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 ML Pipeline", 
        "📊 Model Comparison", 
        "🌊 Data Flow",
        "🎯 Process Details"
    ])
    
    with tab1:
        show_ml_pipeline_animation()
    
    with tab2:
        show_model_comparison_animation()
    
    with tab3:
        show_data_flow_animation()
    
    with tab4:
        show_process_details_animation()

def show_ml_pipeline_animation():
    """Show animated ML pipeline visualization"""
    
    st.markdown("## 🎬 ML Pipeline Animation")
    
    # Pipeline steps
    pipeline_steps = [
        {"name": "Data Collection", "icon": "📊", "color": "#FF6B6B", "duration": 2},
        {"name": "Data Cleaning", "icon": "🧹", "color": "#4ECDC4", "duration": 3},
        {"name": "Feature Engineering", "icon": "⚙️", "color": "#45B7D1", "duration": 2},
        {"name": "Model Training", "icon": "🧠", "color": "#96CEB4", "duration": 4},
        {"name": "Model Evaluation", "icon": "📈", "color": "#FFEAA7", "duration": 2},
        {"name": "Deployment", "icon": "🚀", "color": "#DDA0DD", "duration": 1}
    ]
    
    # Animation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)
    with col2:
        show_details = st.checkbox("Show Details", True)
    with col3:
        auto_play = st.checkbox("Auto Play", False)
    
    if st.button("▶️ Start Animation") or auto_play:
        animate_pipeline(pipeline_steps, animation_speed, show_details)

def show_model_comparison_animation():
    """Show animated model comparison visualization"""
    
    st.markdown("## 📊 Model Performance Comparison")
    st.markdown("Watch how different algorithms perform as they learn from data")
    
    # Model selection
    available_models = ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree"]
    selected_models = st.multiselect(
        "Select models to compare:",
        available_models,
        default=["Logistic Regression", "Random Forest", "SVM"]
    )
    
    if len(selected_models) >= 2:
        # Animation controls
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Training Epochs", 10, 100, 50)
        with col2:
            update_frequency = st.slider("Update Frequency (ms)", 100, 1000, 200)
        
        if st.button("🚀 Start Comparison"):
            animate_model_comparison(selected_models, epochs, update_frequency)
    else:
        st.info("Please select at least 2 models to compare")

def show_process_details_animation():
    """Show detailed process animations"""
    
    st.markdown("## 🎯 Detailed Process Animations")
    st.markdown("Deep dive into specific ML processes with step-by-step animations")
    
    # Process selection
    processes = {
        "🧹 Data Preprocessing": "preprocessing",
        "🔍 Feature Selection": "feature_selection", 
        "🎯 Hyperparameter Tuning": "hyperparameter_tuning",
        "📊 Cross Validation": "cross_validation",
        "🧠 Neural Network Training": "neural_network"
    }
    
    selected_process = st.selectbox(
        "Select process to animate:",
        list(processes.keys())
    )
    
    if st.button("▶️ Animate Process"):
        process_key = processes[selected_process]
        animate_detailed_process(process_key, selected_process)

def animate_model_comparison(models: List[str], epochs: int, update_frequency: int):
    """Animate model comparison over training epochs"""
    
    # Create placeholder for the plot
    placeholder = st.empty()
    
    # Simulate model performance over epochs
    np.random.seed(42)
    
    # Create figure
    fig = go.Figure()
    
    # Colors for models
    colors = px.colors.qualitative.Set3[:len(models)]
    
    for epoch in range(1, epochs + 1):
        fig.data = []  # Clear previous traces
        
        for i, model in enumerate(models):
            # Simulate performance improvement with some noise
            base_performance = 0.5 + (epoch / epochs) * 0.4
            noise = np.random.normal(0, 0.05)
            performance = min(0.95, max(0.3, base_performance + noise))
            
            # Add some model-specific characteristics
            if "Random Forest" in model:
                performance += 0.05  # Generally performs better
            elif "KNN" in model:
                performance -= 0.02  # Slightly lower performance
            
            # Create trace for this model
            x_vals = list(range(1, epoch + 1))
            y_vals = [0.5 + (e / epochs) * 0.4 + np.random.normal(0, 0.02) for e in x_vals]
            y_vals[-1] = performance  # Set current performance
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=model,
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"Model Performance Comparison - Epoch {epoch}/{epochs}",
            xaxis_title="Training Epoch",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.3, 1.0]),
            height=400,
            showlegend=True
        )
        
        with placeholder:
            st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(update_frequency / 1000)
    
    st.success("🎉 Model comparison animation completed!")

def animate_detailed_process(process_key: str, process_name: str):
    """Animate detailed process steps"""
    
    st.markdown(f"### {process_name} Animation")
    
    # Define process steps
    process_steps = {
        "preprocessing": [
            {"step": "Load Raw Data", "description": "Reading data from source", "time": 1},
            {"step": "Handle Missing Values", "description": "Filling or removing missing data", "time": 2},
            {"step": "Remove Outliers", "description": "Detecting and handling outliers", "time": 2},
            {"step": "Encode Categories", "description": "Converting categorical to numeric", "time": 1},
            {"step": "Scale Features", "description": "Normalizing feature ranges", "time": 1}
        ],
        "feature_selection": [
            {"step": "Calculate Feature Importance", "description": "Ranking features by importance", "time": 2},
            {"step": "Remove Low Variance", "description": "Eliminating constant features", "time": 1},
            {"step": "Correlation Analysis", "description": "Finding highly correlated features", "time": 2},
            {"step": "Select Best Features", "description": "Choosing optimal feature subset", "time": 1}
        ],
        "hyperparameter_tuning": [
            {"step": "Define Parameter Grid", "description": "Setting parameter ranges", "time": 1},
            {"step": "Cross Validation Setup", "description": "Preparing validation strategy", "time": 1},
            {"step": "Grid Search", "description": "Testing parameter combinations", "time": 4},
            {"step": "Best Parameters", "description": "Selecting optimal parameters", "time": 1}
        ],
        "cross_validation": [
            {"step": "Split Data into Folds", "description": "Creating k validation folds", "time": 1},
            {"step": "Fold 1 Training", "description": "Training on folds 2-k", "time": 2},
            {"step": "Fold 1 Validation", "description": "Testing on fold 1", "time": 1},
            {"step": "Repeat for All Folds", "description": "Training and testing each fold", "time": 3},
            {"step": "Average Results", "description": "Computing final performance", "time": 1}
        ],
        "neural_network": [
            {"step": "Initialize Weights", "description": "Setting random initial weights", "time": 1},
            {"step": "Forward Propagation", "description": "Computing predictions", "time": 2},
            {"step": "Calculate Loss", "description": "Measuring prediction error", "time": 1},
            {"step": "Backpropagation", "description": "Computing gradients", "time": 2},
            {"step": "Update Weights", "description": "Adjusting model parameters", "time": 1},
            {"step": "Repeat Epochs", "description": "Iterating training process", "time": 2}
        ]
    }
    
    steps = process_steps.get(process_key, [])
    
    if not steps:
        st.warning("Process animation not available yet!")
        return
    
    # Create progress containers
    progress_placeholder = st.empty()
    step_placeholder = st.empty()
    
    total_time = sum(step["time"] for step in steps)
    current_time = 0
    
    for i, step in enumerate(steps):
        # Update progress
        progress = (current_time + step["time"]) / total_time
        
        with progress_placeholder:
            st.progress(progress)
            st.markdown(f"**Step {i+1}/{len(steps)}:** {step['step']}")
        
        with step_placeholder:
            st.info(f"🔄 {step['description']}")
        
        # Simulate step execution time
        for t in range(step["time"] * 10):
            time.sleep(0.1)
            step_progress = (current_time + (t + 1) / 10) / total_time
            with progress_placeholder:
                st.progress(step_progress)
        
        current_time += step["time"]
        
        with step_placeholder:
            st.success(f"✅ {step['step']} completed!")
        
        time.sleep(0.5)
    
    st.balloons()
    st.success(f"🎉 {process_name} animation completed!")

def animate_pipeline(steps: List[Dict], speed: float, show_details: bool):
    """Animate the ML pipeline process"""
    
    # Create containers for animation
    progress_container = st.container()
    details_container = st.container() if show_details else None
    visual_container = st.container()
    
    total_duration = sum(step["duration"] for step in steps)
    
    for i, step in enumerate(steps):
        # Update progress
        progress = (i + 1) / len(steps)
        
        with progress_container:
            st.progress(progress)
            st.markdown(f"### {step['icon']} {step['name']}")
        
        # Show details if enabled
        if show_details and details_container:
            with details_container:
                show_step_details(step, i)
        
        # Create animated visualization
        with visual_container:
            create_step_visualization(step, i, steps)
        
        # Wait based on step duration and speed
        time.sleep(step["duration"] / speed)
    
    # Show completion
    with progress_container:
        st.success("🎉 Pipeline Complete!")
        st.balloons()

def show_step_details(step: Dict, step_index: int):
    """Show detailed information for each pipeline step"""
    
    details = {
        0: {  # Data Collection
            "description": "Gathering data from various sources",
            "tasks": ["Data extraction", "Format validation", "Initial quality checks"],
            "output": "Raw dataset ready for processing"
        },
        1: {  # Data Cleaning
            "description": "Cleaning and preparing the data",
            "tasks": ["Handle missing values", "Remove duplicates", "Fix data types"],
            "output": "Clean, consistent dataset"
        },
        2: {  # Feature Engineering
            "description": "Creating and transforming features",
            "tasks": ["Feature selection", "Encoding", "Scaling", "Creating new features"],
            "output": "Engineered feature set"
        },
        3: {  # Model Training
            "description": "Training machine learning models",
            "tasks": ["Algorithm selection", "Hyperparameter tuning", "Cross-validation"],
            "output": "Trained ML model"
        },
        4: {  # Model Evaluation
            "description": "Evaluating model performance",
            "tasks": ["Metrics calculation", "Validation testing", "Performance analysis"],
            "output": "Model performance report"
        },
        5: {  # Deployment
            "description": "Deploying the model to production",
            "tasks": ["Model packaging", "API creation", "Monitoring setup"],
            "output": "Production-ready ML system"
        }
    }
    
    step_detail = details.get(step_index, {})
    
    st.info(f"📝 **{step_detail.get('description', 'Processing...')}**")
    
    if "tasks" in step_detail:
        st.markdown("**Tasks:**")
        for task in step_detail["tasks"]:
            st.write(f"• {task}")
    
    if "output" in step_detail:
        st.success(f"✅ **Output:** {step_detail['output']}")

def create_step_visualization(step: Dict, step_index: int, all_steps: List[Dict]):
    """Create animated visualization for each step"""
    
    if step_index == 0:  # Data Collection
        show_data_collection_animation()
    elif step_index == 1:  # Data Cleaning
        show_data_cleaning_animation()
    elif step_index == 2:  # Feature Engineering
        show_feature_engineering_animation()
    elif step_index == 3:  # Model Training
        show_model_training_animation()
    elif step_index == 4:  # Model Evaluation
        show_model_evaluation_animation()
    elif step_index == 5:  # Deployment
        show_deployment_animation()

def show_data_collection_animation():
    """Animate data collection process"""
    
    # Create data points flowing in
    fig = go.Figure()
    
    # Simulate data points appearing
    x_data = []
    y_data = []
    
    for i in range(20):
        x_data.append(np.random.uniform(0, 10))
        y_data.append(np.random.uniform(0, 10))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(size=8, color=f'rgba(255, 107, 107, {0.8})', symbol='circle'),
            name='Data Points',
            showlegend=False
        ))
        
        fig.update_layout(
            title="📊 Data Collection in Progress",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=400,
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10])
        )
        
        # Update plot
        placeholder = st.empty()
        with placeholder:
            st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.1)

def show_data_cleaning_animation():
    """Animate data cleaning process"""
    
    # Create messy data that gets cleaned
    np.random.seed(42)
    n_points = 50
    
    # Original messy data
    x_messy = np.random.normal(5, 3, n_points)
    y_messy = np.random.normal(5, 3, n_points)
    
    # Add some outliers and missing values
    outlier_indices = np.random.choice(n_points, 5, replace=False)
    x_messy[outlier_indices] = np.random.uniform(-5, 15, 5)
    y_messy[outlier_indices] = np.random.uniform(-5, 15, 5)
    
    # Clean data (remove outliers, center data)
    x_clean = x_messy[(x_messy > 0) & (x_messy < 10) & (y_messy > 0) & (y_messy < 10)]
    y_clean = y_messy[(x_messy > 0) & (x_messy < 10) & (y_messy > 0) & (y_messy < 10)]
    
    # Animation
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before Cleaning", "After Cleaning")
    )
    
    # Before cleaning
    fig.add_trace(
        go.Scatter(x=x_messy, y=y_messy, mode='markers',
                  marker=dict(color='red', size=8), name='Messy Data'),
        row=1, col=1
    )
    
    # After cleaning
    fig.add_trace(
        go.Scatter(x=x_clean, y=y_clean, mode='markers',
                  marker=dict(color='green', size=8), name='Clean Data'),
        row=1, col=2
    )
    
    fig.update_layout(
        title="🧹 Data Cleaning Process",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_engineering_animation():
    """Animate feature engineering process"""
    
    # Show feature transformation
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_original = x + np.random.normal(0, 1, 100)
    
    # Engineered features
    y_squared = x**2 + np.random.normal(0, 2, 100)
    y_log = np.log(x + 1) * 5 + np.random.normal(0, 0.5, 100)
    
    # Create 2D subplot for the first three plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original Feature", "Squared Feature", "Log Feature", "Feature Correlation")
    )
    
    # Original
    fig.add_trace(
        go.Scatter(x=x, y=y_original, mode='lines+markers', name='Original'),
        row=1, col=1
    )
    
    # Squared
    fig.add_trace(
        go.Scatter(x=x, y=y_squared, mode='lines+markers', name='Squared'),
        row=1, col=2
    )
    
    # Log
    fig.add_trace(
        go.Scatter(x=x, y=y_log, mode='lines+markers', name='Log'),
        row=2, col=1
    )
    
    # Feature correlation (2D scatter plot instead of 3D)
    fig.add_trace(
        go.Scatter(x=y_original, y=y_squared, mode='markers',
                  marker=dict(size=6, color=y_log, colorscale='Viridis', showscale=True,
                             colorbar=dict(title="Log Feature")),
                  name='Feature Correlation'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="⚙️ Feature Engineering in Action",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a separate 3D visualization
    st.markdown("#### 🌟 3D Feature Space Visualization")
    fig_3d = go.Figure(data=[
        go.Scatter3d(
            x=y_original, 
            y=y_squared, 
            z=y_log,
            mode='markers',
            marker=dict(
                size=4,
                color=x,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Original X Value")
            ),
            name='3D Feature Space'
        )
    ])
    
    fig_3d.update_layout(
        title="3D Feature Space Visualization",
        scene=dict(
            xaxis_title='Original Feature',
            yaxis_title='Squared Feature', 
            zaxis_title='Log Feature'
        ),
        height=500
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

def show_model_training_animation():
    """Animate model training process"""
    
    # Show loss decreasing over epochs
    epochs = range(1, 21)
    loss_values = []
    accuracy_values = []
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Loss", "Training Accuracy")
    )
    
    placeholder = st.empty()
    
    for epoch in epochs:
        # Simulate decreasing loss and increasing accuracy
        loss = 2.0 * np.exp(-epoch/10) + np.random.normal(0, 0.1)
        accuracy = 1.0 - np.exp(-epoch/8) + np.random.normal(0, 0.02)
        
        loss_values.append(max(0, loss))
        accuracy_values.append(min(1, max(0, accuracy)))
        
        # Update plots
        fig.data = []  # Clear previous traces
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=list(range(1, len(loss_values)+1)), y=loss_values,
                      mode='lines+markers', name='Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=list(range(1, len(accuracy_values)+1)), y=accuracy_values,
                      mode='lines+markers', name='Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"🧠 Model Training - Epoch {epoch}",
            height=400
        )
        
        with placeholder:
            st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.2)

def show_model_evaluation_animation():
    """Animate model evaluation process"""
    
    # Show confusion matrix building up
    classes = ['Class A', 'Class B', 'Class C']
    confusion_matrix = np.array([
        [45, 3, 2],
        [2, 38, 5],
        [1, 4, 40]
    ])
    
    # Animate confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=classes,
        y=classes,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="📈 Model Evaluation - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics
    metrics = {
        'Accuracy': 0.892,
        'Precision': 0.887,
        'Recall': 0.885,
        'F1-Score': 0.886
    }
    
    col1, col2, col3, col4 = st.columns(4)
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3, col4][i]:
            st.metric(metric, f"{value:.3f}")

def show_deployment_animation():
    """Animate deployment process"""
    
    # Show system architecture
    fig = go.Figure()
    
    # Add nodes for system components
    components = [
        {"name": "User Interface", "x": 1, "y": 3, "color": "#FF6B6B"},
        {"name": "API Gateway", "x": 2, "y": 3, "color": "#4ECDC4"},
        {"name": "ML Model", "x": 3, "y": 3, "color": "#45B7D1"},
        {"name": "Database", "x": 3, "y": 2, "color": "#96CEB4"},
        {"name": "Monitoring", "x": 2, "y": 1, "color": "#FFEAA7"}
    ]
    
    # Add components
    for comp in components:
        fig.add_trace(go.Scatter(
            x=[comp["x"]], y=[comp["y"]],
            mode='markers+text',
            marker=dict(size=50, color=comp["color"]),
            text=comp["name"],
            textposition="middle center",
            name=comp["name"],
            showlegend=False
        ))
    
    # Add connections
    connections = [
        (1, 3, 2, 3),  # UI to API
        (2, 3, 3, 3),  # API to Model
        (3, 3, 3, 2),  # Model to DB
        (2, 3, 2, 1),  # API to Monitoring
    ]
    
    for x1, y1, x2, y2 in connections:
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="🚀 Deployment Architecture",
        xaxis=dict(range=[0, 4], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 4], showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_interactive_model_comparison():
    """Show interactive model comparison with animations"""
    
    st.markdown("## 🏆 Interactive Model Comparison")
    
    # Sample model performance data
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Neural Network']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time']
    
    # Generate sample data
    np.random.seed(42)
    performance_data = {}
    
    for model in models:
        performance_data[model] = {
            'Accuracy': np.random.uniform(0.7, 0.95),
            'Precision': np.random.uniform(0.65, 0.92),
            'Recall': np.random.uniform(0.68, 0.90),
            'F1-Score': np.random.uniform(0.66, 0.91),
            'Training Time': np.random.uniform(0.5, 10.0)
        }
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Select models to compare:",
            models,
            default=models[:3]
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart type:",
            ["Radar Chart", "Bar Chart", "Heatmap", "Parallel Coordinates"]
        )
    
    if selected_models:
        create_comparison_chart(performance_data, selected_models, chart_type, metrics)

def create_comparison_chart(data: Dict, models: List[str], chart_type: str, metrics: List[str]):
    """Create animated comparison charts"""
    
    if chart_type == "Radar Chart":
        fig = go.Figure()
        
        for model in models:
            values = [data[model][metric] for metric in metrics[:-1]]  # Exclude training time
            values += [values[0]]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics[:-1] + [metrics[0]],
                fill='toself',
                name=model,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="Model Performance Radar Chart",
            height=500
        )
    
    elif chart_type == "Bar Chart":
        # Create animated bar chart
        metric_to_show = st.selectbox("Select metric:", metrics)
        
        model_names = models
        values = [data[model][metric_to_show] for model in models]
        
        fig = px.bar(
            x=model_names,
            y=values,
            title=f"Model Comparison - {metric_to_show}",
            color=values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
    
    elif chart_type == "Heatmap":
        # Create heatmap of all metrics
        heatmap_data = []
        for model in models:
            row = [data[model][metric] for metric in metrics]
            heatmap_data.append(row)
        
        fig = px.imshow(
            heatmap_data,
            x=metrics,
            y=models,
            title="Model Performance Heatmap",
            color_continuous_scale='RdYlBu_r',
            text_auto=True
        )
        
        fig.update_layout(height=400)
    
    else:  # Parallel Coordinates
        # Prepare data for parallel coordinates
        parallel_data = []
        for model in models:
            row = {'Model': model}
            for metric in metrics:
                row[metric] = data[model][metric]
            parallel_data.append(row)
        
        df = pd.DataFrame(parallel_data)
        
        fig = px.parallel_coordinates(
            df,
            dimensions=metrics,
            color="Accuracy",
            title="Model Performance Parallel Coordinates"
        )
        
        fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)

def show_data_flow_animation():
    """Show animated data flow through ML pipeline"""
    
    st.markdown("## 🌊 Data Flow Animation")
    
    # Create flowing data visualization
    placeholder = st.empty()
    
    for frame in range(50):
        fig = go.Figure()
        
        # Create flowing particles
        n_particles = 20
        x_positions = []
        y_positions = []
        colors = []
        
        for i in range(n_particles):
            # Calculate position based on frame and particle index
            progress = (frame + i * 2) % 100 / 100
            
            if progress < 0.2:  # Data collection stage
                x = progress * 5 / 0.2
                y = 2 + np.sin(progress * 10) * 0.5
                color = 'red'
            elif progress < 0.4:  # Preprocessing stage
                x = 5 + (progress - 0.2) * 3 / 0.2
                y = 2 + np.cos((progress - 0.2) * 15) * 0.3
                color = 'blue'
            elif progress < 0.6:  # Training stage
                x = 8 + (progress - 0.4) * 2 / 0.2
                y = 2 + (progress - 0.4) * 2 / 0.2
                color = 'green'
            elif progress < 0.8:  # Evaluation stage
                x = 10 + (progress - 0.6) * 2 / 0.2
                y = 4 - (progress - 0.6) * 2 / 0.2
                color = 'orange'
            else:  # Deployment stage
                x = 12 + (progress - 0.8) * 3 / 0.2
                y = 2
                color = 'purple'
            
            x_positions.append(x)
            y_positions.append(y)
            colors.append(color)
        
        # Add particles
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.7
            ),
            name='Data Flow',
            showlegend=False
        ))
        
        # Add stage labels
        stages = [
            ("📊 Collection", 2.5, 3),
            ("🧹 Preprocessing", 6.5, 3),
            ("🧠 Training", 9, 3),
            ("📈 Evaluation", 11, 3),
            ("🚀 Deployment", 13.5, 3)
        ]
        
        for stage_name, x, y in stages:
            fig.add_annotation(
                x=x, y=y,
                text=stage_name,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        fig.update_layout(
            title="🌊 Real-time Data Flow Through ML Pipeline",
            xaxis=dict(range=[0, 15], showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False),
            height=300,
            showlegend=False
        )
        
        with placeholder:
            st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.1)

if __name__ == "__main__":
    show_ml_pipeline_animation()
