"""
Model-related utility functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

def get_available_models() -> Dict[str, Any]:
    """Get dictionary of available models with their classes and default parameters"""
    
    models = {
        "Logistic Regression": {
            "class": LogisticRegression,
            "params": {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42
            },
            "hyperparams": {
                "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
                "max_iter": {"type": "slider", "min": 100, "max": 2000, "default": 1000, "step": 100}
            },
            "description": "Linear model for classification that uses logistic function to model probabilities."
        },
        "Decision Tree": {
            "class": DecisionTreeClassifier,
            "params": {
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42
            },
            "hyperparams": {
                "max_depth": {"type": "slider", "min": 1, "max": 20, "default": 5, "step": 1},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "step": 1},
                "min_samples_leaf": {"type": "slider", "min": 1, "max": 10, "default": 1, "step": 1}
            },
            "description": "Tree-like model that makes decisions by splitting data based on feature values."
        },
        "Random Forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            },
            "hyperparams": {
                "n_estimators": {"type": "slider", "min": 10, "max": 200, "default": 100, "step": 10},
                "max_depth": {"type": "slider", "min": 1, "max": 20, "default": 5, "step": 1},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "step": 1}
            },
            "description": "Ensemble method that combines multiple decision trees to improve accuracy."
        },
        "Support Vector Machine": {
            "class": SVC,
            "params": {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True
            },
            "hyperparams": {
                "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
                "kernel": {"type": "selectbox", "options": ["rbf", "linear", "poly"], "default": "rbf"},
                "gamma": {"type": "selectbox", "options": ["scale", "auto"], "default": "scale"}
            },
            "description": "Finds optimal boundary between classes by maximizing margin in high-dimensional space."
        },
        "K-Nearest Neighbors": {
            "class": KNeighborsClassifier,
            "params": {
                "n_neighbors": 5,
                "weights": "uniform"
            },
            "hyperparams": {
                "n_neighbors": {"type": "slider", "min": 1, "max": 20, "default": 5, "step": 1},
                "weights": {"type": "selectbox", "options": ["uniform", "distance"], "default": "uniform"},
                "algorithm": {"type": "selectbox", "options": ["auto", "ball_tree", "kd_tree", "brute"], "default": "auto"}
            },
            "description": "Classifies based on majority vote of k nearest neighbors in feature space."
        },
        "Naive Bayes": {
            "class": GaussianNB,
            "params": {},
            "hyperparams": {
                "var_smoothing": {"type": "slider", "min": 1e-10, "max": 1e-5, "default": 1e-9, "step": 1e-10}
            },
            "description": "Probabilistic classifier based on Bayes theorem with strong independence assumptions."
        }
    }
    
    return models

def create_hyperparameter_widgets(model_name: str, model_info: Dict) -> Dict[str, Any]:
    """Create Streamlit widgets for hyperparameter tuning"""
    
    st.subheader(f"🎛️ {model_name} Hyperparameters")
    
    params = {}
    hyperparams = model_info.get("hyperparams", {})
    
    for param_name, param_config in hyperparams.items():
        param_type = param_config["type"]
        
        if param_type == "slider":
            if isinstance(param_config["min"], float) or isinstance(param_config["default"], float):
                params[param_name] = st.slider(
                    param_name.replace("_", " ").title(),
                    min_value=param_config["min"],
                    max_value=param_config["max"],
                    value=param_config["default"],
                    step=param_config["step"],
                    help=f"Adjust the {param_name} parameter"
                )
            else:
                params[param_name] = st.slider(
                    param_name.replace("_", " ").title(),
                    min_value=int(param_config["min"]),
                    max_value=int(param_config["max"]),
                    value=int(param_config["default"]),
                    step=int(param_config["step"]),
                    help=f"Adjust the {param_name} parameter"
                )
        
        elif param_type == "selectbox":
            params[param_name] = st.selectbox(
                param_name.replace("_", " ").title(),
                options=param_config["options"],
                index=param_config["options"].index(param_config["default"]),
                help=f"Select the {param_name} parameter"
            )
    
    return params

def train_model(model_class: Any, params: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train a machine learning model"""
    
    try:
        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def evaluate_classification_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a classification model and return metrics"""
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return metrics, y_pred, y_pred_proba
        
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        return {}, None, None

def perform_cross_validation(model_class: Any, params: Dict, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Perform cross-validation and return average scores"""
    
    try:
        # Guard against non-numeric features
        if isinstance(X, pd.DataFrame) and X.select_dtypes(exclude=[np.number]).shape[1] > 0:
            raise ValueError("Non-numeric features detected. Encode categorical variables before cross-validation.")

        model = model_class(**params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std Accuracy": cv_scores.std(),
            "CV Scores": cv_scores.tolist()
        }
        
        return cv_results
        
    except Exception as e:
        st.error(f"Error in cross-validation: {str(e)}")
        return {}

def get_feature_importance(model: Any, feature_names: List[str]) -> Optional[np.ndarray]:
    """Get feature importance from trained model if available"""
    
    try:
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute values of coefficients
            return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return None
    except Exception as e:
        st.warning(f"Could not extract feature importance: {str(e)}")
        return None

def save_model(model: Any, model_name: str, feature_names: List[str], target_name: str) -> str:
    """Save trained model to disk"""
    
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create model metadata
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "target_name": target_name,
            "model_type": type(model).__name__,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save model
        filename = f"{model_name.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        filepath = models_dir / filename
        
        joblib.dump(model_data, filepath)
        
        return str(filepath)
        
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return ""

def load_model(filepath: str) -> Optional[Dict]:
    """Load saved model from disk"""
    
    try:
        model_data = joblib.load(filepath)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_saved_models() -> List[str]:
    """Get list of saved model files"""
    
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.joblib"))
    return [str(f) for f in model_files]

def prepare_data_for_modeling(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple:
    """Prepare data for modeling by splitting into features and target"""
    
    try:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None, None, []

def encode_categorical_variables(X_train: pd.DataFrame, X_test: pd.DataFrame, encoding_method: str = "onehot") -> Tuple:
    """Encode categorical variables in training and test sets"""
    
    try:
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return X_train, X_test, []
        
        if encoding_method == "onehot":
            # One-hot encoding
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            
            # Fit on training data and transform both sets
            X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols])
            X_test_cat_encoded = encoder.transform(X_test[categorical_cols])
            
            # Get feature names
            feature_names = encoder.get_feature_names_out(categorical_cols)
            
            # Create DataFrames with encoded features
            X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=feature_names, index=X_train.index)
            X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=feature_names, index=X_test.index)
            
            # Combine with numerical features
            X_train_encoded = pd.concat([
                X_train.drop(columns=categorical_cols),
                X_train_cat_df
            ], axis=1)
            
            X_test_encoded = pd.concat([
                X_test.drop(columns=categorical_cols),
                X_test_cat_df
            ], axis=1)
            
        elif encoding_method == "label":
            # Label encoding
            X_train_encoded = X_train.copy()
            X_test_encoded = X_test.copy()
            
            encoders = {}
            for col in categorical_cols:
                encoder = LabelEncoder()
                X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))
                
                # Handle unknown categories in test set
                test_values = X_test[col].astype(str)
                mask = test_values.isin(encoder.classes_)
                X_test_encoded[col] = 0  # Default value for unknown categories
                X_test_encoded.loc[mask, col] = encoder.transform(test_values[mask])
                
                encoders[col] = encoder
            
            feature_names = X_train_encoded.columns.tolist()
        
        return X_train_encoded, X_test_encoded, feature_names
        
    except Exception as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        return X_train, X_test, X_train.columns.tolist()

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaling_method: str = "standard") -> Tuple:
    """Scale numerical features in training and test sets"""
    
    try:
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            return X_train, X_test
        
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            return X_train, X_test
        
        # Fit on training data and transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        return X_train_scaled, X_test_scaled
        
    except Exception as e:
        st.error(f"Error scaling features: {str(e)}")
        return X_train, X_test

def handle_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame, strategy: str = "mean") -> Tuple:
    """Handle missing values in training and test sets"""
    
    try:
        # Separate numerical and categorical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()
        
        # Handle numerical columns
        if numerical_cols:
            num_strategy = strategy if strategy in ['mean', 'median'] else 'mean'
            num_imputer = SimpleImputer(strategy=num_strategy)
            
            X_train_imputed[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
            X_test_imputed[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
        
        # Handle categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            X_train_imputed[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
            X_test_imputed[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        
        return X_train_imputed, X_test_imputed
        
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return X_train, X_test
