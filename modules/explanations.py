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
    "Decision Tree": {
        "intuition": "Splits data along feature thresholds to create human-readable rules.",
        "assumptions": [
            "Few monotonic relationships are helpful",
            "No need for feature scaling"
        ],
        "strengths": [
            "Interpretable rules",
            "Captures non-linearities and interactions",
            "Handles mixed feature types"
        ],
        "weaknesses": [
            "Can overfit without constraints",
            "Unstable to small data changes"
        ],
        "tips": [
            "Control depth/min_samples to avoid overfitting",
            "Prefer ensembles (Random Forest) for robustness"
        ]
    },
    "Random Forest": {
        "intuition": "Ensemble of decision trees trained on random subsets of data and features for robust predictions.",
        "assumptions": ["Trees are high-variance; bagging reduces variance"],
        "strengths": [
            "Strong baseline with good accuracy",
            "Robust to noise and outliers",
            "Feature importance available"
        ],
        "weaknesses": [
            "Less interpretable than a single tree",
            "Can be slower and memory heavy"
        ],
        "tips": [
            "Tune n_estimators and max_depth",
            "Use class_weight for imbalance",
            "Not sensitive to feature scaling"
        ]
    },
    "Support Vector Machine": {
        "intuition": "Finds the maximum-margin separating hyperplane; kernels allow non-linear boundaries.",
        "assumptions": [
            "Classes are separable in some (possibly kernelized) space",
            "Feature scaling is important"
        ],
        "strengths": [
            "Strong performance with the right kernel",
            "Effective in high-dimensional spaces"
        ],
        "weaknesses": [
            "Sensitive to hyperparameters",
            "Scaling required; can be slow on large datasets"
        ],
        "tips": [
            "Scale features",
            "Start with RBF kernel and tune C and gamma",
            "Use probability=True if you need probabilities"
        ]
    },
    "K-Nearest Neighbors": {
        "intuition": "Classifies a point based on the majority label among its nearest neighbors.",
        "assumptions": ["Similar points are near each other in feature space"],
        "strengths": [
            "Simple and intuitive",
            "Non-parametric, captures complex shapes"
        ],
        "weaknesses": [
            "Prediction can be slow",
            "Sensitive to feature scaling and irrelevant features"
        ],
        "tips": [
            "Scale features",
            "Tune k (odd numbers for binary)",
            "Use distance weighting when classes overlap"
        ]
    },
    "Naive Bayes": {
        "intuition": "Applies Bayes' theorem with a strong (naive) independence assumption between features.",
        "assumptions": ["Conditional independence of features given the class"],
        "strengths": [
            "Very fast and simple",
            "Works well with text (bag-of-words)"
        ],
        "weaknesses": [
            "Independence rarely holds exactly",
            "Decision boundaries can be too simple"
        ],
        "tips": [
            "Great baseline",
            "Consider TF-IDF for text",
            "Calibrate probabilities if needed"
        ]
    }
}

DATASETS = {
    "Two Moons": {"fn": make_moons, "kwargs": {"noise": 0.25}},
    "Concentric Circles": {"fn": make_circles, "kwargs": {"noise": 0.2, "factor": 0.5}},
    "Linearly Separable": {"fn": make_classification, "kwargs": {"n_features": 2, "n_redundant": 0, "n_informative": 2, "class_sep": 2.0}},
    "Overlapping Blobs": {"fn": make_classification, "kwargs": {"n_features": 2, "n_redundant": 0, "n_informative": 2, "class_sep": 0.6}}
}

# ---------------------
# Plot helpers
# ---------------------

def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary", scaler=None):
    # Prepare grid
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        try:
            grid = scaler.transform(grid)
        except Exception:
            pass

    # If the model expects scaled inputs, we assume training phase used scaling when enabled
    try:
        zz = model.predict(grid)
    except Exception:
        zz = np.zeros(grid.shape[0])
    zz = zz.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 300),
        y=np.linspace(y_min, y_max, 300),
        z=zz,
        colorscale="RdBu",
        showscale=False,
        opacity=0.3,
        contours=dict(showlines=False)
    ))

    # Scatter original points
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                             marker=dict(color=y, colorscale='RdBu', line=dict(width=0.5, color='black')),
                             name='Samples'))

    fig.update_layout(title=title, xaxis_title="Feature 1", yaxis_title="Feature 2", height=500)
    return fig

# ---------------------
# Main UI
# ---------------------

def show():
    st.markdown("## 📚 Algorithm Guide & Interactive Lab")
    st.write("Learn the intuition, trade-offs, and hyperparameters. Then experiment on toy datasets and see decision boundaries.")

    with st.expander("What you'll learn", expanded=True):
        st.markdown(
            "- Intuition and assumptions of common classifiers\n"
            "- Strengths, weaknesses, and when to use each\n"
            "- How hyperparameters shape decision boundaries\n"
            "- Hands-on: tweak and observe effects instantly"
        )

    tabs = st.tabs(["Interactive Lab", "Cheat Sheets", "Overfitting vs. Underfitting", "Resources"])

    # ---------------------
    # Interactive Lab
    # ---------------------
    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            algo_names = list(ALGOS.keys())
            algo = st.selectbox("Algorithm", algo_names, index=0)
            st.caption(ALGOS[algo]["intuition"]) 

            ds_name = st.selectbox("Toy Dataset", list(DATASETS.keys()), index=0)
            n_samples = st.slider("Samples", min_value=100, max_value=2000, value=500, step=50)
            noise = st.slider("Noise", min_value=0.0, max_value=0.6, value=DATASETS[ds_name]["kwargs"].get("noise", 0.2), step=0.02)
            standardize = st.checkbox("Standardize features", value=algo in ["Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression"])

            # Generate data
            gen = DATASETS[ds_name]
            kwargs = dict(gen["kwargs"])  # copy
            if "noise" in kwargs:
                kwargs["noise"] = noise
            X, y = gen["fn"](n_samples=n_samples, random_state=42, **kwargs)

            # Scale if requested
            scaler = None
            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Hyperparameters (reuse model registry)
            models = get_available_models()
            model_info = models[algo]
            st.markdown("### Hyperparameters")
            params = create_hyperparameter_widgets(algo, model_info)

            # Train & evaluate
            model = train_model(model_info["class"], params, pd.DataFrame(X, columns=["x1", "x2"]), pd.Series(y))
            if model is None:
                st.error("Training failed with current parameters. Try adjusting them.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            avg = 'binary'
            if len(np.unique(y)) > 2:
                avg = 'weighted'
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
                "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
                "F1": f1_score(y_test, y_pred, average=avg, zero_division=0)
            }
            st.markdown("### Performance")
            colm = st.columns(4)
            for (k, v), c in zip(metrics.items(), colm):
                with c:
                    st.metric(k, f"{v:.3f}")

        with right:
            st.markdown("### Decision Boundary")
            fig = plot_decision_boundary(model, X, y, title=f"{algo} on {ds_name}", scaler=scaler)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Cheat Sheets
    # ---------------------
    with tabs[1]:
        st.markdown("### Quick Guides")
        for name in ALGOS:
            with st.expander(f"{name}", expanded=False):
                a = ALGOS[name]
                st.markdown(f"**Intuition:** {a['intuition']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strengths**")
                    st.markdown("\n".join([f"- {s}" for s in a["strengths"]]))
                with col2:
                    st.markdown("**Weaknesses**")
                    st.markdown("\n".join([f"- {w}" for w in a["weaknesses"]]))
                st.markdown("**Assumptions**")
                st.markdown("\n".join([f"- {w}" for w in a["assumptions"]]))
                st.markdown("**Tuning Tips**")
                st.markdown("\n".join([f"- {t}" for t in a["tips"]]))

    # ---------------------
    # Overfitting vs. Underfitting
    # ---------------------
    with tabs[2]:
        st.markdown("### Bias-Variance Trade-off")
        st.write("Underfitting has high bias and low variance; overfitting has low bias and high variance. The goal is balance.")
        st.markdown("- Increase model complexity to reduce bias but watch variance\n- Use cross-validation to estimate generalization\n- Regularize (or prune) to combat overfitting\n- Get more data or augment to stabilize high-variance models")

        col1, col2 = st.columns(2)
        with col1:
            st.info("Examples of increasing complexity: deeper trees, higher C for SVM, more neighbors -> less complex for KNN")
        with col2:
            st.info("Regularization examples: tree max_depth/min_samples, Logistic Regression C, dropout/weight decay in deep nets")

    # ---------------------
    # Resources
    # ---------------------
    with tabs[3]:
        st.markdown("### Learn More")
        st.markdown(
            "- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html\n"
            "- Elements of Statistical Learning (free PDF)\n"
            "- Machine Learning Mastery tutorials\n"
            "- Interpretable ML Book: https://christophm.github.io/interpretable-ml-book/\n"
            "- Data Science Handbook"
        )
