"""
Data Preprocessing Module
Comprehensive data preprocessing with encoding, scaling, and imputation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from utils.helpers import (
    get_numerical_columns, get_categorical_columns, show_learning_tip,
    create_missing_value_plot, display_dataframe_info
)
from utils.model_utils import (
    encode_categorical_variables, scale_features, handle_missing_values,
    prepare_data_for_modeling
)

def show():
    """Main function to display the preprocessing interface"""
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' section!")
        return
    
    df = st.session_state.data.copy()
    
    st.title("⚙️ Data Preprocessing Module")
    st.markdown("Prepare your data for machine learning with comprehensive preprocessing tools!")
    
    # Learning tip
    show_learning_tip(
        "Data preprocessing is crucial for ML success. Clean, properly encoded, and scaled data "
        "leads to better model performance and more reliable results.",
        "info"
    )
    
    # Preprocessing steps
    preprocessing_tabs = st.tabs([
        "🎯 Target Selection",
        "❓ Missing Values", 
        "🏷️ Categorical Encoding",
        "📏 Feature Scaling",
        "✂️ Train-Test Split",
        "✅ Final Review"
    ])
    
    with preprocessing_tabs[0]:
        target_col = handle_target_selection(df)
    
    with preprocessing_tabs[1]:
        df = handle_missing_values_ui(df)
    
    with preprocessing_tabs[2]:
        df = handle_categorical_encoding_ui(df)
    
    with preprocessing_tabs[3]:
        df = handle_feature_scaling_ui(df)
    
    with preprocessing_tabs[4]:
        if target_col:
            handle_train_test_split_ui(df, target_col)
    
    with preprocessing_tabs[5]:
        show_preprocessing_summary()

def handle_target_selection(df: pd.DataFrame) -> str:
    """Handle target variable selection"""
    
    st.markdown("## 🎯 Target Variable Selection")
    st.markdown("Choose the variable you want to predict")
    
    # Show current selection if exists
    current_target = st.session_state.get('target_name', '')
    
    if current_target and current_target in df.columns:
        st.success(f"✅ Current target: **{current_target}**")
        change_target = st.checkbox("Change target variable")
        if not change_target:
            return current_target
    
    # Target selection
    target_col = st.selectbox(
        "Select target variable:",
        [""] + list(df.columns),
        index=list(df.columns).index(current_target) + 1 if current_target in df.columns else 0,
        help="Choose the column you want to predict"
    )
    
    if target_col:
        st.session_state.target_name = target_col
        
        # Analyze target variable
        st.markdown(f"### 📊 Target Analysis: {target_col}")
        
        # Determine task type
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
            task_type = "Classification"
            st.info(f"🎯 **Task Type:** {task_type} ({df[target_col].nunique()} classes)")
            
            # Show class distribution
            value_counts = df[target_col].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Class Distribution:**")
                for class_name, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
            
            with col2:
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Target Distribution"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Check for imbalance
            balance_ratio = value_counts.max() / value_counts.min()
            if balance_ratio > 5:
                st.warning(f"⚠️ Imbalanced dataset detected (ratio: {balance_ratio:.1f}:1). "
                          "Consider balancing techniques during model training.")
        
        else:
            task_type = "Regression"
            st.info(f"🎯 **Task Type:** {task_type}")
            
            # Show target statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[target_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[target_col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[target_col].std():.2f}")
            with col4:
                st.metric("Range", f"{df[target_col].max() - df[target_col].min():.2f}")
            
            # Distribution plot
            fig = px.histogram(
                df, 
                x=target_col,
                title=f"Target Distribution: {target_col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return target_col
    
    return ""

def handle_missing_values_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with user interface"""
    
    st.markdown("## ❓ Missing Values Handling")
    st.markdown("Deal with missing data in your dataset")
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0].index.tolist()
    
    if not missing_cols:
        st.success("🎉 No missing values found! Your data is ready to go.")
        return df
    
    st.warning(f"⚠️ Found missing values in {len(missing_cols)} columns")
    
    # Show missing values visualization
    fig = create_missing_value_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values table
    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing Count': missing_summary.values,
        'Missing %': (missing_summary.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    st.dataframe(missing_df, use_container_width=True)
    
    # Handling strategy
    st.markdown("### 🛠️ Choose Handling Strategy")
    
    strategy_options = {
        "Drop rows with missing values": "drop_rows",
        "Drop columns with missing values": "drop_columns",
        "Impute missing values": "impute"
    }
    
    strategy = st.selectbox(
        "Select strategy:",
        list(strategy_options.keys()),
        help="Choose how to handle missing values"
    )
    
    strategy_key = strategy_options[strategy]
    
    if strategy_key == "drop_rows":
        threshold = st.slider(
            "Drop rows with missing values in how many columns?",
            1, len(missing_cols), 1,
            help="Rows with missing values in this many or more columns will be dropped"
        )
        
        if st.button("Apply Row Dropping"):
            original_rows = len(df)
            df_clean = df.dropna(thresh=len(df.columns) - threshold + 1)
            rows_dropped = original_rows - len(df_clean)
            
            st.success(f"✅ Dropped {rows_dropped} rows ({rows_dropped/original_rows*100:.1f}%)")
            st.info(f"Remaining dataset: {len(df_clean)} rows")
            
            return df_clean
    
    elif strategy_key == "drop_columns":
        threshold = st.slider(
            "Drop columns with more than what percentage of missing values?",
            0, 100, 50,
            help="Columns with missing percentage above this will be dropped"
        )
        
        cols_to_drop = missing_df[missing_df['Missing %'] > threshold]['Column'].tolist()
        
        if cols_to_drop:
            st.warning(f"Columns to be dropped: {', '.join(cols_to_drop)}")
            
            if st.button("Apply Column Dropping"):
                df_clean = df.drop(columns=cols_to_drop)
                st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                return df_clean
        else:
            st.info("No columns meet the dropping criteria")
    
    elif strategy_key == "impute":
        st.markdown("#### 🔧 Imputation Settings")
        
        numerical_cols = get_numerical_columns(df)
        categorical_cols = get_categorical_columns(df)
        
        # Numerical imputation
        if any(col in numerical_cols for col in missing_cols):
            num_strategy = st.selectbox(
                "Numerical columns imputation:",
                ["mean", "median", "most_frequent"],
                help="Strategy for filling missing numerical values"
            )
        
        # Categorical imputation
        if any(col in categorical_cols for col in missing_cols):
            cat_strategy = st.selectbox(
                "Categorical columns imputation:",
                ["most_frequent", "constant"],
                help="Strategy for filling missing categorical values"
            )
            
            if cat_strategy == "constant":
                fill_value = st.text_input("Fill value for categorical:", "Unknown")
        
        if st.button("Apply Imputation"):
            df_clean = df.copy()
            
            # Handle numerical columns
            num_missing_cols = [col for col in missing_cols if col in numerical_cols]
            if num_missing_cols:
                imputer = SimpleImputer(strategy=num_strategy)
                df_clean[num_missing_cols] = imputer.fit_transform(df[num_missing_cols])
            
            # Handle categorical columns
            cat_missing_cols = [col for col in missing_cols if col in categorical_cols]
            if cat_missing_cols:
                if cat_strategy == "most_frequent":
                    imputer = SimpleImputer(strategy="most_frequent")
                    df_clean[cat_missing_cols] = imputer.fit_transform(df[cat_missing_cols])
                else:
                    df_clean[cat_missing_cols] = df_clean[cat_missing_cols].fillna(fill_value)
            
            st.success("✅ Imputation completed!")
            
            # Show results
            remaining_missing = df_clean.isnull().sum().sum()
            st.info(f"Remaining missing values: {remaining_missing}")
            
            return df_clean
    
    return df

def handle_categorical_encoding_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Handle categorical encoding with user interface"""
    
    st.markdown("## 🏷️ Categorical Encoding")
    st.markdown("Convert categorical variables to numerical format")
    
    categorical_cols = get_categorical_columns(df)
    # Exclude target from encoding
    target_col = st.session_state.get('target_name', '')
    if target_col in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != target_col]
        st.info(f"Excluding target column '{target_col}' from encoding.")
    
    if not categorical_cols:
        st.success("✅ No categorical columns found. Your data is ready!")
        return df
    
    st.info(f"Found {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
    
    # Show categorical columns info
    cat_info = []
    for col in categorical_cols:
        cat_info.append({
            'Column': col,
            'Unique Values': df[col].nunique(),
            'Sample Values': ', '.join(df[col].dropna().astype(str).unique()[:3])
        })
    
    cat_df = pd.DataFrame(cat_info)
    st.dataframe(cat_df, use_container_width=True)
    
    # Encoding strategy
    st.markdown("### 🛠️ Choose Encoding Strategy")
    
    encoding_method = st.selectbox(
        "Select encoding method:",
        ["One-Hot Encoding", "Label Encoding"],
        help="Choose how to encode categorical variables"
    )
    
    if encoding_method == "One-Hot Encoding":
        st.info("""
        **One-Hot Encoding:**
        - Creates binary columns for each category
        - Best for nominal categories (no order)
        - Can increase dimensionality significantly
        - Recommended for tree-based models
        """)
        
        # Option to select specific columns
        cols_to_encode = st.multiselect(
            "Select columns to encode (leave empty for all):",
            categorical_cols,
            default=categorical_cols
        )
        
        if not cols_to_encode:
            cols_to_encode = categorical_cols
        
        # Warn about high cardinality
        high_card_cols = [col for col in cols_to_encode if df[col].nunique() > 10]
        if high_card_cols:
            st.warning(f"⚠️ High cardinality columns: {', '.join(high_card_cols)}. "
                      "Consider grouping rare categories first.")
        
        if st.button("Apply One-Hot Encoding"):
            try:
                df_encoded = df.copy()
                
                # Apply one-hot encoding
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                
                # Encode selected columns
                encoded_data = encoder.fit_transform(df[cols_to_encode])
                feature_names = encoder.get_feature_names_out(cols_to_encode)
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                
                # Combine with original data (excluding encoded columns)
                df_encoded = pd.concat([
                    df.drop(columns=cols_to_encode),
                    encoded_df
                ], axis=1)
                
                st.success(f"✅ One-hot encoding applied! Created {len(feature_names)} new columns.")
                st.info(f"Dataset shape: {df.shape} → {df_encoded.shape}")
                
                return df_encoded
                
            except Exception as e:
                st.error(f"Error in one-hot encoding: {str(e)}")
    
    elif encoding_method == "Label Encoding":
        st.info("""
        **Label Encoding:**
        - Assigns integers to each category
        - Best for ordinal categories (with order)
        - Maintains original dimensionality
        - May introduce unwanted ordering for nominal variables
        """)
        
        # Option to select specific columns
        cols_to_encode = st.multiselect(
            "Select columns to encode (leave empty for all):",
            categorical_cols,
            default=categorical_cols
        )
        
        if not cols_to_encode:
            cols_to_encode = categorical_cols
        
        if st.button("Apply Label Encoding"):
            try:
                df_encoded = df.copy()
                
                # Apply label encoding
                for col in cols_to_encode:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df[col].astype(str))
                
                st.success(f"✅ Label encoding applied to {len(cols_to_encode)} columns!")
                
                # Show encoding mappings
                with st.expander("View Encoding Mappings"):
                    for col in cols_to_encode:
                        unique_vals = df[col].dropna().unique()
                        encoder = LabelEncoder()
                        encoded_vals = encoder.fit_transform(unique_vals.astype(str))
                        
                        mapping_df = pd.DataFrame({
                            'Original': unique_vals,
                            'Encoded': encoded_vals
                        })
                        st.write(f"**{col}:**")
                        st.dataframe(mapping_df, use_container_width=True)
                
                return df_encoded
                
            except Exception as e:
                st.error(f"Error in label encoding: {str(e)}")
    
    return df

def handle_feature_scaling_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Handle feature scaling with user interface"""
    
    st.markdown("## 📏 Feature Scaling")
    st.markdown("Normalize numerical features to similar scales")
    
    numerical_cols = get_numerical_columns(df)
    
    # Exclude target column from scaling if it exists
    target_col = st.session_state.get('target_name', '')
    if target_col and target_col in numerical_cols:
        numerical_cols = [col for col in numerical_cols if col != target_col]
    
    if not numerical_cols:
        st.success("✅ No numerical columns to scale!")
        return df
    
    st.info(f"Found {len(numerical_cols)} numerical columns to scale")
    
    # Show current scales
    st.markdown("### 📊 Current Feature Scales")
    
    scale_info = []
    for col in numerical_cols:
        scale_info.append({
            'Column': col,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Mean': df[col].mean(),
            'Std': df[col].std()
        })
    
    scale_df = pd.DataFrame(scale_info)
    for col in ['Min', 'Max', 'Mean', 'Std']:
        scale_df[col] = scale_df[col].round(3)
    
    st.dataframe(scale_df, use_container_width=True)
    
    # Check if scaling is needed
    ranges = scale_df['Max'] - scale_df['Min']
    max_range = ranges.max()
    min_range = ranges.min()
    
    if max_range / min_range > 10:
        st.warning("⚠️ Features have very different scales. Scaling is recommended!")
    else:
        st.info("ℹ️ Features have similar scales, but scaling may still be beneficial for some algorithms.")
    
    # Scaling method selection
    st.markdown("### 🛠️ Choose Scaling Method")
    
    scaling_method = st.selectbox(
        "Select scaling method:",
        ["None", "Standard Scaling", "Min-Max Scaling"],
        help="Choose how to scale numerical features"
    )
    
    if scaling_method == "Standard Scaling":
        st.info("""
        **Standard Scaling (Z-score normalization):**
        - Mean = 0, Standard deviation = 1
        - Best for normally distributed features
        - Preserves the shape of the distribution
        - Recommended for algorithms like SVM, Neural Networks
        """)
    elif scaling_method == "Min-Max Scaling":
        st.info("""
        **Min-Max Scaling:**
        - Scales features to range [0, 1]
        - Preserves relationships between values
        - Sensitive to outliers
        - Good for algorithms that need bounded inputs
        """)
    
    if scaling_method != "None":
        # Option to select specific columns
        cols_to_scale = st.multiselect(
            "Select columns to scale (leave empty for all):",
            numerical_cols,
            default=numerical_cols
        )
        
        if not cols_to_scale:
            cols_to_scale = numerical_cols
        
        if st.button(f"Apply {scaling_method}"):
            try:
                df_scaled = df.copy()
                
                if scaling_method == "Standard Scaling":
                    scaler = StandardScaler()
                else:  # Min-Max Scaling
                    scaler = MinMaxScaler()
                
                # Apply scaling
                df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                
                st.success(f"✅ {scaling_method} applied to {len(cols_to_scale)} columns!")
                
                # Show before/after comparison
                st.markdown("#### Before vs After Scaling")
                
                comparison_col = st.selectbox("Select column to compare:", cols_to_scale)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df, 
                        x=comparison_col,
                        title=f"Before Scaling: {comparison_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        df_scaled, 
                        x=comparison_col,
                        title=f"After Scaling: {comparison_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                return df_scaled
                
            except Exception as e:
                st.error(f"Error in scaling: {str(e)}")
    
    return df

def handle_train_test_split_ui(df: pd.DataFrame, target_col: str):
    """Handle train-test split with user interface"""
    
    st.markdown("## ✂️ Train-Test Split")
    st.markdown("Split your data for model training and evaluation")
    
    if not target_col:
        st.warning("⚠️ Please select a target variable first!")
        return
    
    # Split configuration
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%):",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
    
    with col2:
        random_state = st.number_input(
            "Random state (for reproducibility):",
            min_value=0,
            max_value=1000,
            value=42,
            help="Set seed for reproducible splits"
        )
    
    # Stratification option for classification
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
        stratify = st.checkbox(
            "Stratify split (recommended for classification)",
            value=True,
            help="Maintain target distribution in both train and test sets"
        )
    else:
        stratify = False
    
    if st.button("Create Train-Test Split"):
        try:
            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Check if stratification is possible
            can_stratify = stratify
            if stratify:
                # Check if all classes have at least 2 samples
                value_counts = y.value_counts()
                min_class_size = value_counts.min()
                if min_class_size < 2:
                    st.warning(f"⚠️ Some classes have only {min_class_size} sample(s). Disabling stratification.")
                    can_stratify = False
                elif min_class_size * (test_size/100) < 1:
                    st.warning(f"⚠️ Test size too small for stratification. Disabling stratification.")
                    can_stratify = False
            
            # Perform split
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size/100,
                    random_state=random_state,
                    stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size/100,
                    random_state=random_state
                )
            
            # Store in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.processed_data = df
            
            st.success("✅ Train-test split completed!")
            
            # Show split information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))
            
            # Show target distribution in splits (for classification)
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
                st.markdown("#### 🎯 Target Distribution in Splits")
                
                train_dist = y_train.value_counts(normalize=True).sort_index()
                test_dist = y_test.value_counts(normalize=True).sort_index()
                
                dist_df = pd.DataFrame({
                    'Training Set': train_dist,
                    'Test Set': test_dist
                }).fillna(0)
                
                dist_df = (dist_df * 100).round(2)
                st.dataframe(dist_df, use_container_width=True)
            
            st.info("🚀 Ready for model training! Navigate to the 'Model Training' section.")
            
        except Exception as e:
            st.error(f"Error in train-test split: {str(e)}")

def show_preprocessing_summary():
    """Show summary of preprocessing steps"""
    
    st.markdown("## ✅ Preprocessing Summary")
    st.markdown("Review your data preprocessing pipeline")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ No preprocessed data available. Complete the preprocessing steps first.")
        return
    
    df = st.session_state.processed_data
    
    # Dataset overview
    st.markdown("### 📊 Final Dataset Overview")
    display_dataframe_info(df)
    
    # Show first few rows
    st.markdown("### 👀 Preprocessed Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Preprocessing checklist
    st.markdown("### ✅ Preprocessing Checklist")
    
    checklist_items = [
        ("Target Variable Selected", st.session_state.get('target_name') is not None),
        ("Missing Values Handled", df.isnull().sum().sum() == 0),
        ("Categorical Variables Encoded", len(get_categorical_columns(df)) == 0),
        ("Features Scaled", "processed" in str(type(df))),  # Simplified check
        ("Train-Test Split Done", st.session_state.get('X_train') is not None)
    ]
    
    for item, status in checklist_items:
        if status:
            st.success(f"✅ {item}")
        else:
            st.warning(f"⏳ {item}")
    
    # Data quality checks
    st.markdown("### 🔍 Data Quality Checks")
    
    quality_checks = []
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    quality_checks.append(("Infinite Values", inf_count == 0, f"Found {inf_count} infinite values"))
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    quality_checks.append(("Duplicate Rows", dup_count == 0, f"Found {dup_count} duplicate rows"))
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    quality_checks.append(("Constant Columns", len(constant_cols) == 0, f"Found {len(constant_cols)} constant columns"))
    
    for check_name, is_good, message in quality_checks:
        if is_good:
            st.success(f"✅ {check_name}: OK")
        else:
            st.warning(f"⚠️ {check_name}: {message}")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if st.session_state.get('X_train') is not None:
        st.success("🎉 Your data is ready for model training!")
        st.info("""
        **Next Steps:**
        1. Navigate to 'Model Training' to train different algorithms
        2. Experiment with hyperparameters
        3. Compare model performance in 'Model Evaluation'
        4. Make predictions on new data
        """)
    else:
        remaining_steps = [item for item, status in checklist_items if not status]
        if remaining_steps:
            st.info(f"**Complete these steps:** {', '.join([item for item, _ in remaining_steps])}")
    
    # Export preprocessed data
    if st.button("📥 Download Preprocessed Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show()
