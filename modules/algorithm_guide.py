"""
Algorithm Explanations & Interactive Demos
----------------------------------------
A learning-focused module with:
- Plain-English intuition for common classifiers
- Key assumptions, strengths/weaknesses, and when to use
- Hyperparameter cheat-sheets
- Interactive decision boundary demos on synthetic data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.model_utils import get_available_models, create_hyperparameter_widgets, train_model

# ---------------------
# Static knowledge base
# ---------------------
ALGOS = {
    "Logistic Regression": {
        "intuition": "Finds a linear decision boundary by modeling the log-odds of the target as a linear function of features.",
        "assumptions": [
            "Linear relationship between features and log-odds",
            "Low multicollinearity",
            "Reasonably large sample size"
        ],
        "strengths": [
            "Fast, interpretable coefficients",
            "Probabilistic outputs",
            "Works well with large sparse features"
        ],
        "weaknesses": [
            "Struggles with strong non-linearities",
            "May underfit complex relationships"
        ],
        "tips": [
            "Increase max_iter if not converging",
            "Use regularization (C) to control overfitting",
            "Scale features for better optimization"
        ]
    },
    "Random Forest": {
        "intuition": "Builds many decision trees on random subsets of data/features, then averages their predictions.",
        "assumptions": [
            "Trees can overfit individually",
            "Averaging reduces variance"
        ],
        "strengths": [
            "Handles non-linear patterns naturally",
            "Built-in feature importance",
            "Robust to outliers"
        ],
        "weaknesses": [
            "Can overfit with many deep trees",
            "Less interpretable than single tree"
        ],
        "tips": [
            "Start with n_estimators=100",
            "Tune max_depth to control overfitting",
            "Use feature importance for selection"
        ]
    },
    "SVM": {
        "intuition": "Finds the optimal hyperplane that separates classes with maximum margin.",
        "assumptions": [
            "Classes are separable (or use soft margin)",
            "Features are scaled appropriately"
        ],
        "strengths": [
            "Effective in high dimensions",
            "Memory efficient",
            "Versatile with different kernels"
        ],
        "weaknesses": [
            "Slow on large datasets",
            "Sensitive to feature scaling",
            "No probabilistic output"
        ],
        "tips": [
            "Always scale features",
            "Try different kernels (rbf, poly, linear)",
            "Tune C parameter for regularization"
        ]
    },
    "KNN": {
        "intuition": "Classifies based on the majority vote of k nearest neighbors in feature space.",
        "assumptions": [
            "Similar instances have similar labels",
            "Local structure is meaningful"
        ],
        "strengths": [
            "Simple and intuitive",
            "No training period",
            "Adapts to local patterns"
        ],
        "weaknesses": [
            "Sensitive to irrelevant features",
            "Computationally expensive at prediction",
            "Struggles with high dimensions"
        ],
        "tips": [
            "Scale features for distance metrics",
            "Try odd values of k to avoid ties",
            "Use cross-validation to tune k"
        ]
    },
    "Naive Bayes": {
        "intuition": "Uses Bayes' theorem assuming features are conditionally independent given the class.",
        "assumptions": [
            "Features are conditionally independent",
            "Prior probabilities are meaningful"
        ],
        "strengths": [
            "Fast training and prediction",
            "Works well with small datasets",
            "Handles multiple classes naturally"
        ],
        "weaknesses": [
            "Strong independence assumption",
            "Can be outperformed by more complex models"
        ],
        "tips": [
            "Good baseline model",
            "Works well with text classification",
            "Consider feature selection"
        ]
    },
    "Decision Tree": {
        "intuition": "Creates a tree-like model of decisions based on feature values.",
        "assumptions": [
            "Relationships can be captured by splits",
            "Tree structure represents decision logic"
        ],
        "strengths": [
            "Highly interpretable",
            "Handles non-linear patterns",
            "No need for feature scaling"
        ],
        "weaknesses": [
            "Prone to overfitting",
            "Unstable (small data changes affect tree)",
            "Biased toward features with many levels"
        ],
        "tips": [
            "Limit max_depth to prevent overfitting",
            "Use min_samples_split for pruning",
            "Consider ensemble methods"
        ]
    }
}


def show():
    """Main function to display the Algorithm Guide"""
    
    st.title("📚 Algorithm Guide & Interactive Demos")
    st.markdown("Learn how different machine learning algorithms work with interactive explanations and visualizations.")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Algorithm Explanations", 
        "🎮 Interactive Demos", 
        "🔧 Hyperparameter Guide",
        "📊 Performance Comparison"
    ])
    
    with tab1:
        show_algorithm_explanations()
    
    with tab2:
        show_interactive_demos()
    
    with tab3:
        show_hyperparameter_guide()
    
    with tab4:
        show_performance_comparison()


def show_algorithm_explanations():
    """Show detailed explanations of each algorithm"""
    
    st.markdown("### 🧠 Understanding Machine Learning Algorithms")
    st.markdown("Select an algorithm to learn how it works, when to use it, and what to watch out for.")
    
    # Algorithm selector
    selected_algo = st.selectbox(
        "Choose an algorithm to explore:",
        list(ALGOS.keys()),
        help="Each algorithm has different strengths and is suited for different types of problems"
    )
    
    if selected_algo:
        algo_info = ALGOS[selected_algo]
        
        # Main explanation
        st.markdown(f"#### 🎯 {selected_algo}")
        st.info(f"**How it works:** {algo_info['intuition']}")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ✅ Strengths")
            for strength in algo_info['strengths']:
                st.write(f"• {strength}")
            
            st.markdown("##### 📋 Key Assumptions")
            for assumption in algo_info['assumptions']:
                st.write(f"• {assumption}")
        
        with col2:
            st.markdown("##### ⚠️ Limitations")
            for weakness in algo_info['weaknesses']:
                st.write(f"• {weakness}")
            
            st.markdown("##### 💡 Pro Tips")
            for tip in algo_info['tips']:
                st.write(f"• {tip}")


def show_interactive_demos():
    """Show interactive decision boundary demonstrations"""
    
    st.markdown("### 🎮 Interactive Algorithm Demonstrations")
    st.markdown("See how different algorithms create decision boundaries on various dataset types.")
    
    # Dataset selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📊 Dataset Configuration")
        
        dataset_type = st.selectbox(
            "Dataset Type",
            ["Moons", "Circles", "Linear Separable", "Random Clusters"],
            help="Different dataset shapes challenge algorithms differently"
        )
        
        n_samples = st.slider("Number of Samples", 100, 1000, 300, step=50)
        noise = st.slider("Noise Level", 0.0, 0.3, 0.1, step=0.05)
        
        # Algorithm selection for demo
        demo_algorithms = st.multiselect(
            "Algorithms to Compare",
            list(ALGOS.keys()),
            default=["Logistic Regression", "Random Forest", "SVM"],
            help="Select multiple algorithms to see how they compare"
        )
    
    with col2:
        if demo_algorithms:
            # Generate synthetic dataset
            X, y = generate_demo_dataset(dataset_type, n_samples, noise)
            
            # Create visualization
            fig = create_decision_boundary_plot(X, y, demo_algorithms, dataset_type)
            st.plotly_chart(fig, use_container_width=True)
    
    # Algorithm behavior insights
    if demo_algorithms:
        st.markdown("#### 🔍 What You're Seeing")
        
        insights = {
            "Moons": "Non-linear data that tests algorithm flexibility. Linear models struggle while tree-based methods excel.",
            "Circles": "Radial patterns that challenge most algorithms. SVMs with RBF kernels often perform best.",
            "Linear Separable": "Classic linear data where simple algorithms like Logistic Regression shine.",
            "Random Clusters": "Multiple clusters that benefit from algorithms that can handle complex boundaries."
        }
        
        st.info(f"**{dataset_type} Dataset:** {insights.get(dataset_type, 'Explore how algorithms handle this pattern.')}")


def show_hyperparameter_guide():
    """Show hyperparameter tuning guide"""
    
    st.markdown("### 🔧 Hyperparameter Tuning Guide")
    st.markdown("Learn what each hyperparameter does and how to tune them effectively.")
    
    # Algorithm selection
    selected_algo = st.selectbox(
        "Select algorithm for hyperparameter guide:",
        list(ALGOS.keys())
    )
    
    if selected_algo:
        show_algorithm_hyperparameters(selected_algo)


def show_algorithm_hyperparameters(algorithm: str):
    """Show hyperparameter guide for specific algorithm"""
    
    hyperparameter_guides = {
        "Logistic Regression": {
            "C": {
                "description": "Inverse of regularization strength",
                "range": "0.01 to 100",
                "effect": "Higher values = less regularization = more complex model",
                "tuning_tip": "Start with 1.0, decrease if overfitting, increase if underfitting"
            },
            "max_iter": {
                "description": "Maximum number of iterations for convergence",
                "range": "100 to 1000+",
                "effect": "Higher values allow more time to find optimal solution",
                "tuning_tip": "Increase if you get convergence warnings"
            }
        },
        "Random Forest": {
            "n_estimators": {
                "description": "Number of trees in the forest",
                "range": "10 to 500+",
                "effect": "More trees = more stable predictions but longer training",
                "tuning_tip": "Start with 100, increase if you have time and data"
            },
            "max_depth": {
                "description": "Maximum depth of each tree",
                "range": "None, 3 to 20",
                "effect": "Deeper trees can capture more complex patterns but may overfit",
                "tuning_tip": "Start with None, then limit if overfitting occurs"
            },
            "min_samples_split": {
                "description": "Minimum samples required to split an internal node",
                "range": "2 to 20",
                "effect": "Higher values prevent overfitting by requiring more samples per split",
                "tuning_tip": "Increase if your trees are too complex"
            }
        },
        "SVM": {
            "C": {
                "description": "Regularization parameter",
                "range": "0.1 to 100",
                "effect": "Higher values = tighter fit to training data",
                "tuning_tip": "Use cross-validation to find optimal balance"
            },
            "kernel": {
                "description": "Kernel function type",
                "range": "linear, poly, rbf, sigmoid",
                "effect": "Determines the shape of decision boundary",
                "tuning_tip": "Try rbf first, then linear for high-dimensional data"
            },
            "gamma": {
                "description": "Kernel coefficient (for rbf, poly, sigmoid)",
                "range": "0.001 to 1.0",
                "effect": "Higher values = more complex decision boundary",
                "tuning_tip": "Lower values for smoother boundaries"
            }
        },
        "KNN": {
            "n_neighbors": {
                "description": "Number of neighbors to consider",
                "range": "1 to 20+",
                "effect": "Lower values = more complex boundaries, higher values = smoother",
                "tuning_tip": "Use cross-validation; odd numbers avoid ties"
            },
            "weights": {
                "description": "Weight function for neighbors",
                "range": "uniform, distance",
                "effect": "Distance weighting gives closer neighbors more influence",
                "tuning_tip": "Try distance weighting for better performance"
            }
        },
        "Naive Bayes": {
            "alpha": {
                "description": "Additive smoothing parameter",
                "range": "0.01 to 10",
                "effect": "Higher values = more smoothing = less overfitting",
                "tuning_tip": "Start with 1.0, adjust based on validation performance"
            }
        },
        "Decision Tree": {
            "max_depth": {
                "description": "Maximum depth of the tree",
                "range": "None, 3 to 20",
                "effect": "Deeper trees can overfit, shallower trees may underfit",
                "tuning_tip": "Start with no limit, then constrain if overfitting"
            },
            "min_samples_split": {
                "description": "Minimum samples required to split a node",
                "range": "2 to 20",
                "effect": "Higher values create simpler trees",
                "tuning_tip": "Increase to reduce overfitting"
            },
            "min_samples_leaf": {
                "description": "Minimum samples required at a leaf node",
                "range": "1 to 10",
                "effect": "Higher values create simpler trees",
                "tuning_tip": "Increase for smoother decision boundaries"
            }
        }
    }
    
    if algorithm in hyperparameter_guides:
        guide = hyperparameter_guides[algorithm]
        
        st.markdown(f"#### {algorithm} Hyperparameters")
        
        for param, info in guide.items():
            with st.expander(f"🔧 {param}"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Typical Range:** {info['range']}")
                st.markdown(f"**Effect:** {info['effect']}")
                st.success(f"💡 **Tuning Tip:** {info['tuning_tip']}")


def show_performance_comparison():
    """Show performance comparison across algorithms"""
    
    st.markdown("### 📊 Algorithm Performance Comparison")
    st.markdown("Compare how different algorithms perform on various types of datasets.")
    
    # Create performance comparison table
    performance_data = {
        "Algorithm": ["Logistic Regression", "Random Forest", "SVM", "KNN", "Naive Bayes", "Decision Tree"],
        "Linear Data": ["Excellent", "Good", "Excellent", "Good", "Good", "Fair"],
        "Non-linear Data": ["Poor", "Excellent", "Good", "Good", "Fair", "Good"],
        "High Dimensions": ["Good", "Fair", "Excellent", "Poor", "Good", "Fair"],
        "Large Datasets": ["Excellent", "Good", "Poor", "Poor", "Excellent", "Good"],
        "Interpretability": ["High", "Medium", "Low", "High", "High", "Very High"],
        "Training Speed": ["Fast", "Medium", "Slow", "Fast", "Very Fast", "Fast"]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    # Display as interactive table
    st.dataframe(
        df_performance,
        use_container_width=True,
        hide_index=True
    )
    
    # Performance insights
    st.markdown("#### 🎯 When to Use Each Algorithm")
    
    use_cases = {
        "Logistic Regression": "Linear relationships, need probabilities, large datasets, baseline model",
        "Random Forest": "Non-linear data, feature importance needed, robust predictions required",
        "SVM": "High-dimensional data, clear margin separation, small to medium datasets",
        "KNN": "Local patterns important, irregular decision boundaries, simple baseline",
        "Naive Bayes": "Text classification, small datasets, fast predictions needed",
        "Decision Tree": "Need interpretable rules, mixed data types, feature interactions"
    }
    
    for algo, use_case in use_cases.items():
        st.write(f"**{algo}:** {use_case}")


def generate_demo_dataset(dataset_type: str, n_samples: int, noise: float):
    """Generate synthetic dataset for demonstration"""
    
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.6, random_state=42)
    elif dataset_type == "Linear Separable":
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=42
        )
    else:  # Random Clusters
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=2,
            flip_y=noise,
            random_state=42
        )
    
    return X, y


def create_decision_boundary_plot(X, y, algorithms, dataset_type):
    """Create decision boundary visualization"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    n_algorithms = len(algorithms)
    cols = min(3, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=algorithms,
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Get available models
    available_models = get_available_models()
    
    # Create mapping for algorithm names
    algorithm_mapping = {
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest", 
        "SVM": "Support Vector Machine",
        "KNN": "K-Nearest Neighbors",
        "Naive Bayes": "Naive Bayes",
        "Decision Tree": "Decision Tree"
    }
    
    for idx, algorithm in enumerate(algorithms):
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Map algorithm name to model utils name
        model_key = algorithm_mapping.get(algorithm, algorithm)
        
        if model_key in available_models:
            try:
                # Get model class and create instance
                model_info = available_models[model_key]
                model_class = model_info["class"]
                model_params = model_info["params"]
                model = model_class(**model_params)
                
                if algorithm == "SVM" or model_key == "Support Vector Machine":
                    # SVM needs scaled data
                    model.fit(X_train_scaled, y_train)
                    mesh_points = np.c_[xx.ravel(), yy.ravel()]
                    mesh_points_scaled = scaler.transform(mesh_points)
                    Z = model.predict(mesh_points_scaled)
                    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
                else:
                    model.fit(X_train, y_train)
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                
                Z = Z.reshape(xx.shape)
                
                # Add contour for decision boundary
                fig.add_trace(
                    go.Contour(
                        x=xx[0],
                        y=yy[:, 0],
                        z=Z,
                        showscale=False,
                        opacity=0.3,
                        line=dict(width=0),
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
                
                # Add data points
                colors = ['red' if label == 0 else 'blue' for label in y]
                fig.add_trace(
                    go.Scatter(
                        x=X[:, 0],
                        y=X[:, 1],
                        mode='markers',
                        marker=dict(
                            color=colors,
                            size=6,
                            line=dict(width=1, color='white')
                        ),
                        name=f'{algorithm} (Acc: {accuracy:.2f})',
                        showlegend=False,
                        hovertemplate=f'{algorithm}<br>Accuracy: {accuracy:.2f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
            except Exception as e:
                st.error(f"Error with {algorithm}: {str(e)}")
    
    # Update layout
    fig.update_layout(
        title=f"Decision Boundaries on {dataset_type} Dataset",
        height=300 * rows,
        showlegend=False
    )
    
    return fig


if __name__ == "__main__":
    show()
