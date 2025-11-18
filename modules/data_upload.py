"""
Data Upload Module for ML Dashboard
Handles file uploads, data preview, and basic metadata display
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
from utils.helpers import display_dataframe_info, show_learning_tip, get_sample_data_description

def show():
    """Main function to display the data upload interface"""
    
    st.title("📁 Data Upload Module")
    st.markdown("Upload your dataset to begin your machine learning journey!")
    
    # Learning tip
    show_learning_tip(
        "Start with a clean, well-structured dataset. CSV and Excel formats work best. "
        "Make sure your target variable (what you want to predict) is clearly defined.",
        "info"
    )
    
    # Upload options
    upload_tab, sample_tab = st.tabs(["📤 Upload Your Data", "🎯 Sample Datasets"])
    
    with upload_tab:
        handle_file_upload()
    
    with sample_tab:
        load_sample_datasets()
    
    # Display loaded data
    if st.session_state.data is not None:
        display_loaded_data()
    
    # Data requirements section
    with st.expander("📋 Data Requirements & Tips"):
        st.markdown("""
        ### ✅ What makes a good dataset for ML?
        
        **Structure Requirements:**
        - Rows represent individual samples/observations
        - Columns represent features/variables
        - One column should be your target variable (what you want to predict)
        - At least 50+ rows for meaningful results (more is better!)
        - Minimal missing values (we can help clean this up later)
        
        **File Format:**
        - CSV (.csv) - Most common and recommended
        - Excel (.xlsx, .xls) - Also supported
        - First row should contain column headers
        
        **Data Quality:**
        - Consistent data types in each column
        - No completely empty rows or columns
        - Target variable should be clearly labeled
        - Features should be relevant to your prediction goal
        
        **Examples of Good Datasets:**
        - Customer data with purchase behavior (predict: will buy/not buy)
        - Student records with grades (predict: pass/fail)
        - House features with prices (predict: house price)
        - Medical symptoms with diagnosis (predict: disease type)
        """)

def handle_file_upload():
    """Handle file upload functionality"""
    
    st.markdown("### 📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing your dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            st.markdown("### 📋 File Information")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Load data based on file type
            with st.spinner("Loading your data..."):
                if uploaded_file.name.endswith('.csv'):
                    df = load_csv_file(uploaded_file)
                else:
                    df = load_excel_file(uploaded_file)
                
                if df is not None:
                    st.session_state.data = df
                    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
                    
                    # Show quick preview
                    st.markdown("### 👀 Quick Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.markdown("""
            **Common issues and solutions:**
            - **Encoding problems**: Try saving your CSV with UTF-8 encoding
            - **Excel issues**: Make sure the file isn't corrupted or password-protected
            - **Large files**: Consider reducing file size or sampling your data
            - **Format issues**: Ensure the first row contains column headers
            """)

def load_csv_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load CSV file with encoding detection"""
    
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.info(f"📝 File loaded with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None
    
    st.error("❌ Could not read the CSV file with any supported encoding")
    return None

def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load Excel file"""
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        
        # If multiple sheets, let user choose
        if len(excel_file.sheet_names) > 1:
            sheet_name = st.selectbox(
                "Select sheet to load:",
                excel_file.sheet_names,
                help="Your Excel file has multiple sheets. Select the one containing your data."
            )
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(uploaded_file)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def load_sample_datasets():
    """Load sample datasets for learning and experimentation"""
    
    st.markdown("### 🎯 Try Sample Datasets")
    st.markdown("Perfect for learning! These datasets are pre-loaded and ready to use.")
    
    # Sample dataset options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌸 Iris Dataset", use_container_width=True):
            load_iris_dataset()
        
        st.markdown(get_sample_data_description("iris"))
    
    with col2:
        if st.button("🚢 Titanic Dataset", use_container_width=True):
            load_titanic_dataset()
        
        st.markdown(get_sample_data_description("titanic"))
    
    # Wine dataset
    if st.button("🍷 Wine Quality Dataset", use_container_width=True):
        load_wine_dataset()
    
    st.markdown("""
    **Wine Quality Dataset**: Contains physicochemical properties of wine samples 
    with quality ratings. Great for both classification and regression tasks!
    """)

def load_iris_dataset():
    """Load the Iris dataset"""
    
    try:
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        st.session_state.data = df
        st.success("✅ Iris dataset loaded successfully!")
        
        # Show dataset info
        st.info("""
        🌸 **Iris Dataset Loaded!**
        - **Features**: 4 (sepal length, sepal width, petal length, petal width)
        - **Target**: species (setosa, versicolor, virginica)
        - **Task**: Multi-class classification
        - **Samples**: 150 (50 per class)
        """)
        
    except ImportError:
        st.error("❌ Scikit-learn not available. Please install it to use sample datasets.")
    except Exception as e:
        st.error(f"❌ Error loading Iris dataset: {str(e)}")

def load_titanic_dataset():
    """Load a simplified Titanic dataset"""
    
    try:
        # Create a simplified Titanic dataset
        np.random.seed(42)
        n_samples = 891
        
        # Generate synthetic Titanic-like data
        data = {
            'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'age': np.random.normal(29, 14, n_samples).clip(0, 80),
            'sibsp': np.random.poisson(0.5, n_samples),
            'parch': np.random.poisson(0.4, n_samples),
            'fare': np.random.lognormal(3.2, 1.3, n_samples),
            'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Generate survival based on realistic probabilities
        survival_prob = 0.2  # Base survival rate
        survival_prob += (df['sex'] == 'female') * 0.5  # Women more likely to survive
        survival_prob += (df['pclass'] == 1) * 0.3  # First class more likely
        survival_prob += (df['pclass'] == 2) * 0.1  # Second class somewhat more likely
        survival_prob -= (df['age'] > 60) * 0.2  # Elderly less likely
        survival_prob += (df['sibsp'] + df['parch'] > 0) * 0.1  # Family members help
        
        df['survived'] = np.random.binomial(1, survival_prob.clip(0, 1), n_samples)
        
        # Add some missing values for realism
        missing_age_idx = np.random.choice(df.index, size=int(0.2 * n_samples), replace=False)
        df.loc[missing_age_idx, 'age'] = np.nan
        
        missing_embarked_idx = np.random.choice(df.index, size=2, replace=False)
        df.loc[missing_embarked_idx, 'embarked'] = np.nan
        
        st.session_state.data = df
        st.success("✅ Titanic dataset loaded successfully!")
        
        st.info("""
        🚢 **Titanic Dataset Loaded!**
        - **Features**: 7 (passenger class, sex, age, siblings/spouses, parents/children, fare, port of embarkation)
        - **Target**: survived (0 = No, 1 = Yes)
        - **Task**: Binary classification
        - **Samples**: 891
        - **Note**: This is a simplified synthetic version for learning purposes
        """)
        
    except Exception as e:
        st.error(f"❌ Error loading Titanic dataset: {str(e)}")

def load_wine_dataset():
    """Load the Wine Quality dataset"""
    
    try:
        # Create a synthetic wine quality dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'fixed_acidity': np.random.normal(7.5, 1.2, n_samples).clip(4, 12),
            'volatile_acidity': np.random.normal(0.4, 0.2, n_samples).clip(0.1, 1.0),
            'citric_acid': np.random.normal(0.3, 0.15, n_samples).clip(0, 0.8),
            'residual_sugar': np.random.lognormal(0.8, 0.7, n_samples).clip(0.5, 15),
            'chlorides': np.random.normal(0.08, 0.03, n_samples).clip(0.01, 0.3),
            'free_sulfur_dioxide': np.random.normal(30, 15, n_samples).clip(1, 80),
            'total_sulfur_dioxide': np.random.normal(120, 40, n_samples).clip(10, 300),
            'density': np.random.normal(0.996, 0.002, n_samples).clip(0.99, 1.01),
            'pH': np.random.normal(3.2, 0.2, n_samples).clip(2.8, 3.8),
            'sulphates': np.random.normal(0.6, 0.2, n_samples).clip(0.3, 1.2),
            'alcohol': np.random.normal(10.5, 1.5, n_samples).clip(8, 15)
        }
        
        df = pd.DataFrame(data)
        
        # Generate quality based on chemical properties
        quality_score = (
            (df['alcohol'] - 8) * 0.3 +
            (12 - df['fixed_acidity']) * 0.1 +
            (0.8 - df['volatile_acidity']) * 2 +
            (df['citric_acid']) * 1.5 +
            (0.15 - df['chlorides']) * 5 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convert to quality scale 3-9
        df['quality'] = (quality_score * 2 + 6).round().clip(3, 9).astype(int)
        
        st.session_state.data = df
        st.success("✅ Wine Quality dataset loaded successfully!")
        
        st.info("""
        🍷 **Wine Quality Dataset Loaded!**
        - **Features**: 11 physicochemical properties
        - **Target**: quality (rating from 3-9)
        - **Task**: Multi-class classification or regression
        - **Samples**: 1000
        - **Note**: Synthetic dataset based on real wine chemistry patterns
        """)
        
    except Exception as e:
        st.error(f"❌ Error loading Wine dataset: {str(e)}")

def display_loaded_data():
    """Display information about the loaded dataset"""
    
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview")
    
    df = st.session_state.data
    
    # Basic info
    display_dataframe_info(df)
    
    # Data preview with options
    st.markdown("### 👀 Data Preview")
    
    preview_options = st.columns([1, 1, 2])
    
    with preview_options[0]:
        num_rows = st.slider("Rows to display", 5, min(50, len(df)), 10)
    
    with preview_options[1]:
        show_info = st.checkbox("Show data types", value=True)
    
    # Display data
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    if show_info:
        st.markdown("### 📋 Column Details")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Data validation
    validate_dataset(df)
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    st.info("""
    Great! Your data is loaded. Here's what you can do next:
    
    1. **🔍 Explore your data** - Go to 'Exploratory Data Analysis' to understand your data better
    2. **⚙️ Preprocess** - Clean and prepare your data for modeling
    3. **🧠 Train models** - Try different algorithms to find the best one
    4. **📈 Evaluate** - Compare model performance and understand results
    """)

def validate_dataset(df: pd.DataFrame):
    """Validate the dataset and provide feedback"""
    
    st.markdown("### ✅ Dataset Validation")
    
    validation_results = []
    
    # Check dataset size
    if len(df) < 50:
        validation_results.append(("⚠️", "Small dataset", f"Only {len(df)} rows. Consider getting more data for better results."))
    elif len(df) < 200:
        validation_results.append(("🟡", "Medium dataset", f"{len(df)} rows. Good for learning, but more data would be better."))
    else:
        validation_results.append(("✅", "Good dataset size", f"{len(df)} rows. Excellent for machine learning!"))
    
    # Check number of features
    if len(df.columns) < 3:
        validation_results.append(("⚠️", "Few features", f"Only {len(df.columns)} columns. More features might improve predictions."))
    else:
        validation_results.append(("✅", "Good feature count", f"{len(df.columns)} columns available for analysis."))
    
    # Check for missing values
    missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percent == 0:
        validation_results.append(("✅", "No missing values", "Perfect! No data cleaning needed."))
    elif missing_percent < 5:
        validation_results.append(("🟡", "Few missing values", f"{missing_percent:.1f}% missing. Easy to handle."))
    elif missing_percent < 20:
        validation_results.append(("⚠️", "Some missing values", f"{missing_percent:.1f}% missing. Will need preprocessing."))
    else:
        validation_results.append(("❌", "Many missing values", f"{missing_percent:.1f}% missing. Significant preprocessing needed."))
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        validation_results.append(("✅", "No duplicates", "No duplicate rows found."))
    else:
        validation_results.append(("⚠️", "Duplicate rows", f"{duplicates} duplicate rows found. Consider removing them."))
    
    # Display results
    for icon, title, message in validation_results:
        st.write(f"{icon} **{title}**: {message}")
    
    # Overall assessment
    warnings = sum(1 for icon, _, _ in validation_results if icon in ["⚠️", "❌"])
    if warnings == 0:
        st.success("🎉 Excellent! Your dataset looks great for machine learning!")
    elif warnings <= 2:
        st.info("👍 Good dataset! Minor issues can be addressed in preprocessing.")
    else:
        st.warning("⚡ Dataset needs attention. Review the issues above before proceeding.")

if __name__ == "__main__":
    show()
