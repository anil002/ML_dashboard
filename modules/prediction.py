"""
Prediction Module
Make predictions on new data with trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import show_learning_tip, get_numerical_columns, get_categorical_columns

def show():
    """Main function to display the prediction interface"""
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        return
    
    st.title("🔮 Prediction Module")
    st.markdown("Make predictions on new data using your trained model!")
    
    # Learning tip
    show_learning_tip(
        "Always ensure new data follows the same preprocessing steps as your training data. "
        "Check data types, scaling, and encoding to get accurate predictions.",
        "info"
    )
    
    # Prediction methods
    prediction_tabs = st.tabs([
        "🎯 Single Prediction",
        "📊 Batch Prediction", 
        "📈 Prediction Analysis",
        "🔍 What-If Analysis"
    ])
    
    with prediction_tabs[0]:
        show_single_prediction()
    
    with prediction_tabs[1]:
        show_batch_prediction()
    
    with prediction_tabs[2]:
        show_prediction_analysis()
    
    with prediction_tabs[3]:
        show_what_if_analysis()

def show_single_prediction():
    """Interface for making single predictions"""
    
    st.markdown("## 🎯 Single Prediction")
    st.markdown("Enter values for individual features to get a prediction")
    
    model = st.session_state.model
    feature_names = st.session_state.feature_names
    model_name = st.session_state.model_name
    target_name = st.session_state.target_name
    
    st.info(f"**Model**: {model_name} | **Target**: {target_name}")
    
    # Create input widgets for each feature
    st.markdown("### 📝 Enter Feature Values")
    
    feature_values = {}
    
    # Get sample data for reference
    X_train = st.session_state.X_train
    
    # Create input widgets based on feature types
    cols = st.columns(2)
    col_idx = 0
    
    for feature in feature_names:
        with cols[col_idx % 2]:
            feature_values[feature] = create_feature_input_widget(feature, X_train[feature])
        col_idx += 1
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        make_single_prediction(feature_values)

def create_feature_input_widget(feature_name: str, feature_data: pd.Series):
    """Create appropriate input widget for feature"""
    
    # Determine feature type and create appropriate widget
    if feature_data.dtype in ['int64', 'float64']:
        # Numerical feature
        min_val = float(feature_data.min())
        max_val = float(feature_data.max())
        mean_val = float(feature_data.mean())
        
        # Use appropriate widget based on value range
        if feature_data.dtype == 'int64' and max_val - min_val < 20:
            # Small integer range - use slider
            value = st.slider(
                feature_name,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(mean_val),
                help=f"Range: {min_val} to {max_val}"
            )
        else:
            # Large range or float - use number input
            value = st.number_input(
                feature_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Range: {min_val:.2f} to {max_val:.2f}"
            )
    
    else:
        # Categorical feature
        unique_values = feature_data.dropna().unique()
        most_common = feature_data.mode().iloc[0] if len(feature_data.mode()) > 0 else unique_values[0]
        
        value = st.selectbox(
            feature_name,
            options=unique_values,
            index=list(unique_values).index(most_common) if most_common in unique_values else 0,
            help=f"Choose from {len(unique_values)} options"
        )
    
    return value

def make_single_prediction(feature_values: dict):
    """Make prediction using input feature values"""
    
    try:
        # Create DataFrame from input values
        input_df = pd.DataFrame([feature_values])
        
        # Make prediction
        model = st.session_state.model
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎉 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Predicted {st.session_state.target_name}**: {prediction}")
            
            if prediction_proba is not None:
                # Get class labels
                classes = model.classes_ if hasattr(model, 'classes_') else [0, 1]
                
                st.markdown("### 📊 Prediction Confidence")
                for i, (class_label, prob) in enumerate(zip(classes, prediction_proba)):
                    confidence_percent = prob * 100
                    st.write(f"**{class_label}**: {confidence_percent:.1f}%")
                    
                    # Progress bar for confidence
                    st.progress(prob)
        
        with col2:
            # Input summary
            st.markdown("### 📝 Input Summary")
            for feature, value in feature_values.items():
                st.write(f"**{feature}**: {value}")
        
        # Confidence visualization
        if prediction_proba is not None and len(prediction_proba) > 1:
            display_prediction_confidence(prediction_proba, classes)
        
        # Feature contribution (if available)
        show_feature_contribution(input_df, feature_values)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Make sure all feature values are entered correctly and match the training data format.")

def display_prediction_confidence(prediction_proba: np.ndarray, classes: list):
    """Display prediction confidence visualization"""
    
    st.markdown("### 📊 Prediction Confidence Visualization")
    
    # Create confidence plot
    fig = px.bar(
        x=[str(class_label) for class_label in classes],
        y=prediction_proba * 100,
        title="Prediction Confidence by Class",
        labels={'x': 'Class', 'y': 'Confidence (%)'},
        color=prediction_proba,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_feature_contribution(input_df: pd.DataFrame, feature_values: dict):
    """Show how each feature contributes to the prediction"""
    
    st.markdown("### 🔍 Feature Contribution Analysis")
    
    # Get feature importance if available
    if st.session_state.feature_importance is not None:
        feature_importance = st.session_state.feature_importance
        feature_names = st.session_state.feature_names
        
        # Create contribution analysis
        contribution_data = []
        
        for i, feature in enumerate(feature_names):
            importance = feature_importance[i]
            value = feature_values.get(feature, 0)
            
            # Normalize input value for visualization
            X_train = st.session_state.X_train
            if feature in X_train.columns:
                if X_train[feature].dtype in ['int64', 'float64']:
                    feature_range = X_train[feature].max() - X_train[feature].min()
                    normalized_value = (value - X_train[feature].min()) / feature_range if feature_range > 0 else 0.5
                else:
                    normalized_value = 0.5  # Default for categorical
            else:
                normalized_value = 0.5
            
            contribution_data.append({
                'Feature': feature,
                'Importance': importance,
                'Input Value': value,
                'Normalized Value': normalized_value,
                'Contribution Score': importance * normalized_value
            })
        
        contribution_df = pd.DataFrame(contribution_data)
        contribution_df = contribution_df.sort_values('Contribution Score', ascending=False)
        
        # Visualize top contributing features
        top_features = contribution_df.head(10)
        
        fig = px.bar(
            top_features,
            x='Contribution Score',
            y='Feature',
            orientation='h',
            title="Top 10 Feature Contributions to This Prediction",
            labels={'Contribution Score': 'Contribution to Prediction'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature contribution table
        with st.expander("📋 Detailed Feature Contributions"):
            st.dataframe(contribution_df[['Feature', 'Input Value', 'Importance', 'Contribution Score']], use_container_width=True)

def show_batch_prediction():
    """Interface for batch predictions on uploaded data"""
    
    st.markdown("## 📊 Batch Prediction")
    st.markdown("Upload a file to make predictions on multiple samples")
    
    model_name = st.session_state.model_name
    target_name = st.session_state.target_name
    feature_names = st.session_state.feature_names
    
    st.info(f"**Model**: {model_name} | **Expected features**: {len(feature_names)}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with the same features as your training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            new_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {new_data.shape}")
            
            # Show preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(new_data.head(), use_container_width=True)
            
            # Validate data
            validation_result = validate_prediction_data(new_data, feature_names)
            
            if validation_result['valid']:
                # Make predictions
                if st.button("🚀 Generate Predictions"):
                    make_batch_predictions(new_data)
            else:
                st.error("❌ Data validation failed:")
                for error in validation_result['errors']:
                    st.write(f"• {error}")
                
                show_data_preparation_help()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def validate_prediction_data(new_data: pd.DataFrame, expected_features: list) -> dict:
    """Validate that new data is compatible with trained model"""
    
    errors = []
    
    # Check if all required features are present
    missing_features = set(expected_features) - set(new_data.columns)
    if missing_features:
        errors.append(f"Missing features: {', '.join(missing_features)}")
    
    # Check for extra features
    extra_features = set(new_data.columns) - set(expected_features)
    if extra_features:
        st.warning(f"⚠️ Extra features will be ignored: {', '.join(extra_features)}")
    
    # Check data types (basic validation)
    X_train = st.session_state.X_train
    for feature in expected_features:
        if feature in new_data.columns and feature in X_train.columns:
            train_dtype = X_train[feature].dtype
            new_dtype = new_data[feature].dtype
            
            # Check for major type mismatches
            if train_dtype in ['int64', 'float64'] and new_dtype not in ['int64', 'float64']:
                errors.append(f"Feature '{feature}' should be numerical, got {new_dtype}")
    
    # Check for completely empty data
    if new_data.empty:
        errors.append("Uploaded data is empty")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def make_batch_predictions(new_data: pd.DataFrame):
    """Make predictions on batch data"""
    
    try:
        model = st.session_state.model
        feature_names = st.session_state.feature_names
        target_name = st.session_state.target_name
        
        # Select and reorder features to match training data
        prediction_data = new_data[feature_names].copy()
        
        # Make predictions
        with st.spinner("Making predictions..."):
            predictions = model.predict(prediction_data)
            
            # Get prediction probabilities if available
            prediction_probabilities = None
            if hasattr(model, 'predict_proba'):
                prediction_probabilities = model.predict_proba(prediction_data)
        
        # Add predictions to the data
        result_data = new_data.copy()
        result_data[f'Predicted_{target_name}'] = predictions
        
        # Add confidence scores if available
        if prediction_probabilities is not None:
            if len(model.classes_) == 2:
                # Binary classification - add probability of positive class
                result_data['Prediction_Confidence'] = prediction_probabilities[:, 1]
            else:
                # Multi-class - add max probability
                result_data['Prediction_Confidence'] = np.max(prediction_probabilities, axis=1)
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎉 Batch Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(predictions))
        with col2:
            if hasattr(model, 'classes_'):
                unique_predictions = pd.Series(predictions).nunique()
                st.metric("Unique Predictions", unique_predictions)
        with col3:
            if prediction_probabilities is not None:
                avg_confidence = np.mean(np.max(prediction_probabilities, axis=1))
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Show results table
        st.markdown("### 📋 Prediction Results")
        st.dataframe(result_data, use_container_width=True)
        
        # Prediction summary
        show_batch_prediction_summary(predictions, prediction_probabilities)
        
        # Download results
        csv = result_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

def show_batch_prediction_summary(predictions: np.ndarray, prediction_probabilities: np.ndarray = None):
    """Show summary of batch predictions"""
    
    st.markdown("### 📊 Prediction Summary")
    
    # Prediction distribution
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution plot
        fig = px.bar(
            x=prediction_counts.index.astype(str),
            y=prediction_counts.values,
            title="Prediction Distribution",
            labels={'x': 'Predicted Class', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution (if available)
        if prediction_probabilities is not None:
            confidence_scores = np.max(prediction_probabilities, axis=1)
            
            fig = px.histogram(
                x=confidence_scores,
                title="Prediction Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'},
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📈 Summary Statistics")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.write("**Prediction Counts:**")
        for pred_class, count in prediction_counts.items():
            percentage = (count / len(predictions)) * 100
            st.write(f"• **{pred_class}**: {count} ({percentage:.1f}%)")
    
    with summary_col2:
        if prediction_probabilities is not None:
            confidence_scores = np.max(prediction_probabilities, axis=1)
            st.write("**Confidence Statistics:**")
            st.write(f"• **Mean**: {np.mean(confidence_scores):.3f}")
            st.write(f"• **Median**: {np.median(confidence_scores):.3f}")
            st.write(f"• **Min**: {np.min(confidence_scores):.3f}")
            st.write(f"• **Max**: {np.max(confidence_scores):.3f}")

def show_data_preparation_help():
    """Show help for preparing data for predictions"""
    
    with st.expander("📋 Data Preparation Guide"):
        st.markdown("""
        ### 🔧 How to Prepare Your Data for Predictions
        
        **Required Format:**
        - CSV file with column headers
        - Same feature names as training data
        - Same data types as training data
        
        **Feature Requirements:**
        """)
        
        feature_names = st.session_state.feature_names
        X_train = st.session_state.X_train
        
        for feature in feature_names:
            if feature in X_train.columns:
                dtype = X_train[feature].dtype
                sample_values = X_train[feature].dropna().head(3).tolist()
                
                st.write(f"• **{feature}** ({dtype}): Examples: {sample_values}")
        
        st.markdown("""
        **Common Issues:**
        - Missing column headers
        - Different column names (case-sensitive)
        - Wrong data types (text instead of numbers)
        - Missing required features
        - Extra spaces in values
        
        **Tips:**
        - Use the exact same column names as training data
        - Ensure numerical features contain only numbers
        - Check for typos in categorical values
        - Remove any target column from prediction data
        """)

def show_prediction_analysis():
    """Show analysis of prediction patterns"""
    
    st.markdown("## 📈 Prediction Analysis")
    st.markdown("Analyze patterns and insights from your model's predictions")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No predictions available. Make some predictions first!")
        return
    
    y_test = st.session_state.y_test
    y_pred = st.session_state.predictions
    y_pred_proba = st.session_state.prediction_probabilities
    
    # Prediction accuracy analysis
    st.markdown("### 🎯 Prediction Accuracy Analysis")
    
    correct_predictions = (y_test == y_pred).sum()
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correct Predictions", correct_predictions)
    with col2:
        st.metric("Total Predictions", total_predictions)
    with col3:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    # Prediction confidence analysis
    if y_pred_proba is not None:
        show_confidence_analysis(y_test, y_pred, y_pred_proba)
    
    # Error analysis
    show_error_analysis(y_test, y_pred)

def show_confidence_analysis(y_test, y_pred, y_pred_proba):
    """Analyze prediction confidence patterns"""
    
    st.markdown("### 🔍 Prediction Confidence Analysis")
    
    # Get confidence scores
    confidence_scores = np.max(y_pred_proba, axis=1)
    correct_mask = (y_test == y_pred)
    
    # Confidence vs accuracy
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution for correct vs incorrect predictions
        confidence_data = []
        
        for is_correct, label in [(True, 'Correct'), (False, 'Incorrect')]:
            mask = correct_mask == is_correct
            if mask.sum() > 0:
                for conf in confidence_scores[mask]:
                    confidence_data.append({'Confidence': conf, 'Prediction': label})
        
        if confidence_data:
            conf_df = pd.DataFrame(confidence_data)
            
            fig = px.histogram(
                conf_df,
                x='Confidence',
                color='Prediction',
                title='Confidence Distribution: Correct vs Incorrect',
                nbins=20,
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average confidence by correctness
        avg_conf_correct = np.mean(confidence_scores[correct_mask])
        avg_conf_incorrect = np.mean(confidence_scores[~correct_mask]) if (~correct_mask).sum() > 0 else 0
        
        conf_comparison = pd.DataFrame({
            'Prediction Type': ['Correct', 'Incorrect'],
            'Average Confidence': [avg_conf_correct, avg_conf_incorrect]
        })
        
        fig = px.bar(
            conf_comparison,
            x='Prediction Type',
            y='Average Confidence',
            title='Average Confidence by Prediction Correctness',
            color='Average Confidence',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence insights
    st.markdown("#### 💡 Confidence Insights")
    
    # High confidence incorrect predictions
    high_conf_threshold = 0.8
    high_conf_incorrect = ((confidence_scores > high_conf_threshold) & (~correct_mask)).sum()
    
    if high_conf_incorrect > 0:
        st.warning(f"⚠️ {high_conf_incorrect} high-confidence incorrect predictions found. "
                  "These might indicate systematic model errors.")
    
    # Low confidence correct predictions
    low_conf_threshold = 0.6
    low_conf_correct = ((confidence_scores < low_conf_threshold) & correct_mask).sum()
    
    if low_conf_correct > 0:
        st.info(f"ℹ️ {low_conf_correct} low-confidence correct predictions. "
               "Model is uncertain but still correct.")

def show_error_analysis(y_test, y_pred):
    """Analyze prediction errors"""
    
    st.markdown("### ❌ Error Analysis")
    
    # Find incorrect predictions
    incorrect_mask = y_test != y_pred
    num_errors = incorrect_mask.sum()
    
    if num_errors == 0:
        st.success("🎉 Perfect predictions! No errors found.")
        return
    
    st.write(f"Found {num_errors} incorrect predictions out of {len(y_test)} total.")
    
    # Error breakdown by class
    error_breakdown = []
    
    for true_class in sorted(y_test.unique()):
        class_mask = y_test == true_class
        class_errors = (incorrect_mask & class_mask).sum()
        class_total = class_mask.sum()
        error_rate = (class_errors / class_total) * 100 if class_total > 0 else 0
        
        error_breakdown.append({
            'True Class': true_class,
            'Errors': class_errors,
            'Total': class_total,
            'Error Rate (%)': error_rate
        })
    
    error_df = pd.DataFrame(error_breakdown)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(error_df, use_container_width=True)
    
    with col2:
        fig = px.bar(
            error_df,
            x='True Class',
            y='Error Rate (%)',
            title='Error Rate by True Class',
            color='Error Rate (%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_what_if_analysis():
    """Show what-if analysis interface"""
    
    st.markdown("## 🔍 What-If Analysis")
    st.markdown("Explore how changing feature values affects predictions")
    
    if st.session_state.feature_importance is None:
        st.warning("⚠️ Feature importance not available for what-if analysis.")
        return
    
    feature_names = st.session_state.feature_names
    feature_importance = st.session_state.feature_importance
    
    # Select feature to analyze
    st.markdown("### 🎛️ Select Feature to Analyze")
    
    # Sort features by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    selected_feature = st.selectbox(
        "Choose feature to vary:",
        importance_df['feature'].tolist(),
        help="Select a feature to see how changing its value affects predictions"
    )
    
    if selected_feature:
        perform_what_if_analysis(selected_feature)

def perform_what_if_analysis(selected_feature: str):
    """Perform what-if analysis for selected feature"""
    
    st.markdown(f"### 📊 What-If Analysis: {selected_feature}")
    
    model = st.session_state.model
    X_train = st.session_state.X_train
    feature_names = st.session_state.feature_names
    
    # Get a representative sample point
    sample_idx = st.slider(
        "Select sample point:",
        0, len(X_train) - 1, 0,
        help="Choose a sample from training data as baseline"
    )
    
    baseline_sample = X_train.iloc[sample_idx].copy()
    
    # Show baseline prediction
    baseline_pred = model.predict([baseline_sample])[0]
    baseline_proba = None
    
    if hasattr(model, 'predict_proba'):
        baseline_proba = model.predict_proba([baseline_sample])[0]
    
    st.write(f"**Baseline prediction**: {baseline_pred}")
    
    if baseline_proba is not None:
        st.write(f"**Baseline confidence**: {np.max(baseline_proba):.3f}")
    
    # Create range of values for selected feature
    feature_data = X_train[selected_feature]
    
    if feature_data.dtype in ['int64', 'float64']:
        # Numerical feature
        min_val = float(feature_data.min())
        max_val = float(feature_data.max())
        
        # Create range of values
        if feature_data.dtype == 'int64':
            test_values = np.arange(int(min_val), int(max_val) + 1)
        else:
            test_values = np.linspace(min_val, max_val, 50)
        
        # Test predictions across range
        predictions = []
        confidences = []
        
        for test_val in test_values:
            test_sample = baseline_sample.copy()
            test_sample[selected_feature] = test_val
            
            pred = model.predict([test_sample])[0]
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([test_sample])[0]
                confidences.append(np.max(proba))
        
        # Visualize results
        what_if_df = pd.DataFrame({
            'Feature_Value': test_values,
            'Prediction': predictions,
            'Confidence': confidences if confidences else [0] * len(predictions)
        })
        
        # Prediction vs feature value
        fig1 = px.line(
            what_if_df,
            x='Feature_Value',
            y='Prediction',
            title=f'Prediction vs {selected_feature}',
            markers=True
        )
        
        # Add baseline point
        fig1.add_vline(
            x=baseline_sample[selected_feature],
            line_dash="dash",
            line_color="red",
            annotation_text="Baseline"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Confidence vs feature value
        if confidences:
            fig2 = px.line(
                what_if_df,
                x='Feature_Value',
                y='Confidence',
                title=f'Prediction Confidence vs {selected_feature}',
                markers=True
            )
            
            fig2.add_vline(
                x=baseline_sample[selected_feature],
                line_dash="dash",
                line_color="red",
                annotation_text="Baseline"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        # Categorical feature
        unique_values = feature_data.unique()
        
        predictions = []
        confidences = []
        
        for test_val in unique_values:
            test_sample = baseline_sample.copy()
            test_sample[selected_feature] = test_val
            
            pred = model.predict([test_sample])[0]
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([test_sample])[0]
                confidences.append(np.max(proba))
        
        # Visualize results
        what_if_df = pd.DataFrame({
            'Feature_Value': unique_values,
            'Prediction': predictions,
            'Confidence': confidences if confidences else [0] * len(predictions)
        })
        
        # Bar chart for categorical
        fig = px.bar(
            what_if_df,
            x='Feature_Value',
            y='Prediction',
            title=f'Prediction vs {selected_feature}',
            color='Confidence' if confidences else None
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("#### 💡 Key Insights")
    
    unique_predictions = len(set(predictions))
    
    if unique_predictions == 1:
        st.info(f"🔒 Changing **{selected_feature}** doesn't affect the prediction for this sample.")
    else:
        st.success(f"🔄 **{selected_feature}** influences predictions - {unique_predictions} different outcomes possible.")
        
        # Find most influential values
        if feature_data.dtype in ['int64', 'float64']:
            # Find values that change prediction from baseline
            baseline_pred_value = baseline_pred
            different_pred_mask = [p != baseline_pred_value for p in predictions]
            
            if any(different_pred_mask):
                changing_values = [test_values[i] for i, changed in enumerate(different_pred_mask) if changed]
                if changing_values:
                    st.write(f"**Values that change prediction**: {min(changing_values):.2f} to {max(changing_values):.2f}")

if __name__ == "__main__":
    show()
