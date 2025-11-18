"""
Exploratory Data Analysis (EDA) Module
Comprehensive data exploration with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helpers import (
    display_dataframe_info, create_distribution_plot, create_correlation_heatmap,
    create_pairplot, get_numerical_columns, get_categorical_columns,
    create_missing_value_plot, show_learning_tip
)

def show():
    """Main function to display the EDA interface"""
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' section!")
        return
    
    df = st.session_state.data
    
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Understand your data through comprehensive analysis and visualizations!")
    
    # Learning tip
    show_learning_tip(
        "EDA is crucial for understanding your data patterns, distributions, and relationships. "
        "It helps you make informed decisions about preprocessing and model selection.",
        "info"
    )
    
    # Main EDA sections
    eda_tabs = st.tabs([
        "📊 Dataset Overview", 
        "📈 Univariate Analysis", 
        "🔗 Bivariate Analysis",
        "🌐 Multivariate Analysis",
        "❓ Missing Values",
        "🎯 Target Analysis"
    ])
    
    with eda_tabs[0]:
        show_dataset_overview(df)
    
    with eda_tabs[1]:
        show_univariate_analysis(df)
    
    with eda_tabs[2]:
        show_bivariate_analysis(df)
    
    with eda_tabs[3]:
        show_multivariate_analysis(df)
    
    with eda_tabs[4]:
        show_missing_values_analysis(df)
    
    with eda_tabs[5]:
        show_target_analysis(df)

def show_dataset_overview(df: pd.DataFrame):
    """Display comprehensive dataset overview"""
    
    st.markdown("## 📊 Dataset Overview")
    
    # Basic statistics
    display_dataframe_info(df)
    
    # Data types breakdown
    st.markdown("### 🏷️ Data Types Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count by data type
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values, 
            names=dtype_counts.index.astype(str),
            title="Column Data Types"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Numerical vs Categorical
        numerical_cols = get_numerical_columns(df)
        categorical_cols = get_categorical_columns(df)
        
        type_summary = {
            "Type": ["Numerical", "Categorical"],
            "Count": [len(numerical_cols), len(categorical_cols)]
        }
        
        fig = px.bar(
            type_summary, 
            x="Type", 
            y="Count",
            title="Numerical vs Categorical Features",
            color="Type"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📋 Summary Statistics")
    
    if len(numerical_cols) > 0:
        st.markdown("#### Numerical Features")
        summary_stats = df[numerical_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Features")
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Most Frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A",
                "Frequency": df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        cat_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_df, use_container_width=True)

def show_univariate_analysis(df: pd.DataFrame):
    """Analyze individual variables"""
    
    st.markdown("## 📈 Univariate Analysis")
    st.markdown("Examine the distribution of individual variables")
    
    numerical_cols = get_numerical_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Numerical variables analysis
    if len(numerical_cols) > 0:
        st.markdown("### 🔢 Numerical Variables")
        
        selected_num_col = st.selectbox(
            "Select numerical column to analyze:",
            numerical_cols,
            key="univar_num"
        )
        
        if selected_num_col:
            analyze_numerical_variable(df, selected_num_col)
    
    # Categorical variables analysis
    if len(categorical_cols) > 0:
        st.markdown("### 🏷️ Categorical Variables")
        
        selected_cat_col = st.selectbox(
            "Select categorical column to analyze:",
            categorical_cols,
            key="univar_cat"
        )
        
        if selected_cat_col:
            analyze_categorical_variable(df, selected_cat_col)

def analyze_numerical_variable(df: pd.DataFrame, column: str):
    """Analyze a numerical variable in detail"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Basic statistics
    with col1:
        st.metric("Mean", f"{df[column].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[column].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[column].std():.2f}")
    with col4:
        st.metric("Range", f"{df[column].max() - df[column].min():.2f}")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Distribution plot
        plot_type = st.selectbox(
            "Select plot type:",
            ["histogram", "box", "violin"],
            key=f"plot_type_{column}"
        )
        
        fig = create_distribution_plot(df, column, plot_type)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Q-Q plot for normality check
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(df[column].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {column}")
        st.pyplot(fig)
        plt.close()
    
    # Distribution analysis
    st.markdown(f"#### 📊 Distribution Analysis for {column}")
    
    # Check for outliers using IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.write(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        st.write(f"**Lower bound:** {lower_bound:.2f}")
        st.write(f"**Upper bound:** {upper_bound:.2f}")
    
    with analysis_col2:
        # Skewness and kurtosis
        skewness = df[column].skew()
        kurtosis = df[column].kurtosis()
        
        st.write(f"**Skewness:** {skewness:.2f}")
        if abs(skewness) < 0.5:
            st.write("✅ Distribution is approximately symmetric")
        elif skewness > 0.5:
            st.write("➡️ Distribution is right-skewed")
        else:
            st.write("⬅️ Distribution is left-skewed")
        
        st.write(f"**Kurtosis:** {kurtosis:.2f}")
        if abs(kurtosis) < 3:
            st.write("📊 Normal-like tail behavior")
        else:
            st.write("📈 Heavy-tailed distribution")

def analyze_categorical_variable(df: pd.DataFrame, column: str):
    """Analyze a categorical variable in detail"""
    
    value_counts = df[column].value_counts()
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Values", df[column].nunique())
    with col2:
        st.metric("Most Frequent", value_counts.index[0])
    with col3:
        st.metric("Frequency", value_counts.iloc[0])
    with col4:
        st.metric("Percentage", f"{value_counts.iloc[0]/len(df)*100:.1f}%")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {column}",
            labels={'x': column, 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Pie chart
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Proportion of {column}"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Value counts table
    st.markdown(f"#### 📋 Value Counts for {column}")
    
    value_summary = pd.DataFrame({
        'Value': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / len(df) * 100).round(2)
    })
    
    st.dataframe(value_summary, use_container_width=True)
    
    # Analysis insights
    if df[column].nunique() > 10:
        st.warning(f"⚠️ High cardinality: {df[column].nunique()} unique values. Consider grouping rare categories.")
    
    # Check for imbalanced categories
    min_freq = value_counts.min()
    max_freq = value_counts.max()
    imbalance_ratio = max_freq / min_freq
    
    if imbalance_ratio > 10:
        st.warning(f"⚠️ Imbalanced categories detected. Ratio: {imbalance_ratio:.1f}:1")

def show_bivariate_analysis(df: pd.DataFrame):
    """Analyze relationships between pairs of variables"""
    
    st.markdown("## 🔗 Bivariate Analysis")
    st.markdown("Explore relationships between pairs of variables")
    
    numerical_cols = get_numerical_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = list(df.columns)
    
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Numerical vs Numerical", "Numerical vs Categorical", "Categorical vs Categorical"]
    )
    
    if analysis_type == "Numerical vs Numerical" and len(numerical_cols) >= 2:
        show_numerical_vs_numerical(df, numerical_cols)
    elif analysis_type == "Numerical vs Categorical" and len(numerical_cols) > 0 and len(categorical_cols) > 0:
        show_numerical_vs_categorical(df, numerical_cols, categorical_cols)
    elif analysis_type == "Categorical vs Categorical" and len(categorical_cols) >= 2:
        show_categorical_vs_categorical(df, categorical_cols)
    else:
        st.warning("⚠️ Not enough variables of the selected types for this analysis.")

def show_numerical_vs_numerical(df: pd.DataFrame, numerical_cols: list):
    """Analyze relationship between numerical variables"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X variable:", numerical_cols, key="x_num")
    with col2:
        y_var = st.selectbox("Select Y variable:", [col for col in numerical_cols if col != x_var], key="y_num")
    
    if x_var and y_var:
        # Scatter plot
        fig = px.scatter(
            df, 
            x=x_var, 
            y=y_var,
            title=f"{x_var} vs {y_var}",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        correlation = df[x_var].corr(df[y_var])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correlation", f"{correlation:.3f}")
        
        with col2:
            if abs(correlation) < 0.3:
                strength = "Weak"
            elif abs(correlation) < 0.7:
                strength = "Moderate"
            else:
                strength = "Strong"
            st.metric("Strength", strength)
        
        with col3:
            direction = "Positive" if correlation > 0 else "Negative"
            st.metric("Direction", direction)
        
        # Interpretation
        st.markdown("#### 🔍 Interpretation")
        if abs(correlation) > 0.7:
            st.success(f"Strong {direction.lower()} correlation detected!")
        elif abs(correlation) > 0.3:
            st.info(f"Moderate {direction.lower()} correlation found.")
        else:
            st.warning("Weak correlation - variables may not be linearly related.")

def show_numerical_vs_categorical(df: pd.DataFrame, numerical_cols: list, categorical_cols: list):
    """Analyze relationship between numerical and categorical variables"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_var = st.selectbox("Select numerical variable:", numerical_cols, key="num_cat")
    with col2:
        cat_var = st.selectbox("Select categorical variable:", categorical_cols, key="cat_num")
    
    if num_var and cat_var:
        # Box plot
        fig = px.box(
            df, 
            x=cat_var, 
            y=num_var,
            title=f"{num_var} by {cat_var}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Violin plot
        fig2 = px.violin(
            df, 
            x=cat_var, 
            y=num_var,
            title=f"Distribution of {num_var} by {cat_var}",
            box=True
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary statistics by category
        st.markdown("#### 📊 Summary Statistics by Category")
        summary_by_cat = df.groupby(cat_var)[num_var].agg(['count', 'mean', 'median', 'std']).round(2)
        st.dataframe(summary_by_cat, use_container_width=True)

def show_categorical_vs_categorical(df: pd.DataFrame, categorical_cols: list):
    """Analyze relationship between categorical variables"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat_var1 = st.selectbox("Select first categorical variable:", categorical_cols, key="cat1")
    with col2:
        cat_var2 = st.selectbox("Select second categorical variable:", 
                               [col for col in categorical_cols if col != cat_var1], key="cat2")
    
    if cat_var1 and cat_var2:
        # Crosstab
        crosstab = pd.crosstab(df[cat_var1], df[cat_var2])
        
        # Heatmap
        fig = px.imshow(
            crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            title=f"Cross-tabulation: {cat_var1} vs {cat_var2}",
            text_auto=True,
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stacked bar chart
        crosstab_pct = pd.crosstab(df[cat_var1], df[cat_var2], normalize='index') * 100
        
        fig2 = go.Figure()
        for col in crosstab_pct.columns:
            fig2.add_trace(go.Bar(
                name=str(col),
                x=crosstab_pct.index,
                y=crosstab_pct[col]
            ))
        
        fig2.update_layout(
            barmode='stack',
            title=f"Percentage Distribution: {cat_var1} vs {cat_var2}",
            xaxis_title=cat_var1,
            yaxis_title="Percentage"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chi-square test
        try:
            from scipy.stats import chi2_contingency
            chi2, p_value, dof, expected = chi2_contingency(crosstab)
            
            st.markdown("#### 🧪 Statistical Test Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chi-square", f"{chi2:.3f}")
            with col2:
                st.metric("p-value", f"{p_value:.3f}")
            with col3:
                significance = "Significant" if p_value < 0.05 else "Not significant"
                st.metric("Result", significance)
            
            if p_value < 0.05:
                st.success("✅ Variables are significantly associated!")
            else:
                st.info("ℹ️ No significant association found.")
                
        except Exception as e:
            st.warning(f"Could not perform chi-square test: {str(e)}")

def show_multivariate_analysis(df: pd.DataFrame):
    """Analyze relationships among multiple variables"""
    
    st.markdown("## 🌐 Multivariate Analysis")
    st.markdown("Explore complex relationships among multiple variables")
    
    numerical_cols = get_numerical_columns(df)
    
    if len(numerical_cols) < 2:
        st.warning("⚠️ Need at least 2 numerical columns for multivariate analysis.")
        return
    
    # Correlation matrix
    st.markdown("### 🔗 Correlation Matrix")
    
    if len(numerical_cols) > 2:
        selected_cols = st.multiselect(
            "Select columns for correlation analysis:",
            numerical_cols,
            default=numerical_cols[:min(8, len(numerical_cols))],
            help="Select up to 8 columns for better visualization"
        )
    else:
        selected_cols = numerical_cols
    
    if len(selected_cols) >= 2:
        # Correlation heatmap
        fig = create_correlation_heatmap(df[selected_cols])
        st.plotly_chart(fig, use_container_width=True)
        
        # Find strong correlations
        corr_matrix = df[selected_cols].corr()
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if strong_corr:
            st.markdown("#### 🎯 Strong Correlations (|r| > 0.7)")
            for col1, col2, corr in strong_corr:
                st.write(f"**{col1}** ↔ **{col2}**: {corr:.3f}")
    
    # Pairplot
    if len(numerical_cols) >= 2:
        st.markdown("### 🎨 Pairplot")
        
        show_pairplot = st.checkbox("Show pairplot (may take time for large datasets)")
        
        if show_pairplot:
            # Limit columns for performance
            pairplot_cols = selected_cols[:min(5, len(selected_cols))]
            if len(pairplot_cols) < len(selected_cols):
                st.warning(f"Showing pairplot for first {len(pairplot_cols)} columns only.")
            
            fig = create_pairplot(df, None, len(pairplot_cols))
            st.plotly_chart(fig, use_container_width=True)

def show_missing_values_analysis(df: pd.DataFrame):
    """Analyze missing values in the dataset"""
    
    st.markdown("## ❓ Missing Values Analysis")
    st.markdown("Understand patterns and extent of missing data")
    
    # Missing values overview
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    if missing_count.sum() == 0:
        st.success("🎉 Congratulations! No missing values found in your dataset!")
        return
    
    # Missing values visualization
    fig = create_missing_value_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values table
    st.markdown("### 📋 Missing Values Summary")
    
    missing_summary = pd.DataFrame({
        'Column': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percentage': missing_percent.values.round(2)
    })
    
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing Percentage', ascending=False)
    
    st.dataframe(missing_summary, use_container_width=True)
    
    # Missing value patterns
    if len(missing_summary) > 1:
        st.markdown("### 🔍 Missing Value Patterns")
        
        # Create missing value pattern matrix
        missing_pattern = df.isnull()
        
        # Show pattern combinations
        pattern_counts = missing_pattern.value_counts()
        st.write(f"**Number of different missing patterns:** {len(pattern_counts)}")
        
        if len(pattern_counts) <= 10:
            st.markdown("#### Top Missing Patterns")
            for i, (pattern, count) in enumerate(pattern_counts.head().items()):
                missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
                if missing_cols:
                    st.write(f"**Pattern {i+1}** ({count} rows): Missing {', '.join(missing_cols)}")
                else:
                    st.write(f"**Pattern {i+1}** ({count} rows): No missing values")
    
    # Recommendations
    st.markdown("### 💡 Handling Recommendations")
    
    for _, row in missing_summary.iterrows():
        col_name = row['Column']
        missing_pct = row['Missing Percentage']
        
        with st.expander(f"Recommendations for {col_name} ({missing_pct:.1f}% missing)"):
            if missing_pct < 5:
                st.success("✅ Low missing percentage - safe to impute or drop rows")
                st.write("**Recommended actions:**")
                st.write("- Drop rows with missing values")
                st.write("- Impute with mean/median (numerical) or mode (categorical)")
            elif missing_pct < 20:
                st.warning("⚠️ Moderate missing percentage - careful imputation needed")
                st.write("**Recommended actions:**")
                st.write("- Advanced imputation (KNN, iterative)")
                st.write("- Analyze missing pattern - is it random?")
                st.write("- Consider creating 'missing' indicator variable")
            else:
                st.error("❌ High missing percentage - consider dropping column")
                st.write("**Recommended actions:**")
                st.write("- Drop the column if not critical")
                st.write("- Collect more data if possible")
                st.write("- Use advanced imputation only if column is very important")

def show_target_analysis(df: pd.DataFrame):
    """Analyze potential target variables"""
    
    st.markdown("## 🎯 Target Variable Analysis")
    st.markdown("Identify and analyze your target variable for modeling")
    
    # Target variable selection
    st.markdown("### 🎯 Select Target Variable")
    
    target_col = st.selectbox(
        "Choose your target variable (what you want to predict):",
        [""] + list(df.columns),
        help="Select the column you want to predict"
    )
    
    if not target_col:
        st.info("👆 Please select a target variable to analyze")
        return
    
    # Store target for later use
    st.session_state.target_name = target_col
    
    st.success(f"✅ Target variable selected: **{target_col}**")
    
    # Target variable analysis
    st.markdown(f"### 📊 Analysis of {target_col}")
    
    # Determine if classification or regression
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
        task_type = "Classification"
        analyze_classification_target(df, target_col)
    else:
        task_type = "Regression"
        analyze_regression_target(df, target_col)
    
    st.info(f"📝 **Detected task type:** {task_type}")
    
    # Feature vs target analysis
    st.markdown("### 🔍 Features vs Target Analysis")
    
    other_cols = [col for col in df.columns if col != target_col]
    
    if other_cols:
        feature_col = st.selectbox(
            "Select a feature to analyze against target:",
            other_cols
        )
        
        if feature_col:
            analyze_feature_vs_target(df, feature_col, target_col, task_type)

def analyze_classification_target(df: pd.DataFrame, target_col: str):
    """Analyze classification target variable"""
    
    value_counts = df[target_col].value_counts()
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Classes", len(value_counts))
    with col2:
        st.metric("Most Frequent", value_counts.index[0])
    with col3:
        st.metric("Majority %", f"{value_counts.iloc[0]/len(df)*100:.1f}%")
    with col4:
        balance_ratio = value_counts.max() / value_counts.min()
        st.metric("Imbalance Ratio", f"{balance_ratio:.1f}:1")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Bar chart
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f"Class Distribution - {target_col}",
            labels={'x': 'Count', 'y': target_col}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Pie chart
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Class Proportions - {target_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Balance analysis
    st.markdown("#### ⚖️ Class Balance Analysis")
    
    if balance_ratio <= 2:
        st.success("✅ Well-balanced classes")
    elif balance_ratio <= 5:
        st.warning("⚠️ Moderately imbalanced - consider balancing techniques")
    else:
        st.error("❌ Highly imbalanced - definitely need balancing techniques")
        
        st.markdown("**Recommended techniques for imbalanced data:**")
        st.write("- SMOTE (Synthetic Minority Oversampling)")
        st.write("- Random undersampling of majority class")
        st.write("- Cost-sensitive learning")
        st.write("- Ensemble methods (Random Forest, XGBoost)")

def analyze_regression_target(df: pd.DataFrame, target_col: str):
    """Analyze regression target variable"""
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{df[target_col].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[target_col].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[target_col].std():.2f}")
    with col4:
        st.metric("Range", f"{df[target_col].max() - df[target_col].min():.2f}")
    
    # Distribution analysis
    fig = create_distribution_plot(df, target_col, "histogram")
    st.plotly_chart(fig, use_container_width=True)
    
    # Check for outliers
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    
    st.markdown("#### 🎯 Target Variable Quality")
    
    outlier_col1, outlier_col2 = st.columns(2)
    
    with outlier_col1:
        st.write(f"**Outliers:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        if len(outliers) > len(df) * 0.1:
            st.warning("⚠️ High number of outliers - consider investigation")
    
    with outlier_col2:
        skewness = df[target_col].skew()
        st.write(f"**Skewness:** {skewness:.2f}")
        if abs(skewness) > 1:
            st.warning("⚠️ Highly skewed - consider transformation")

def analyze_feature_vs_target(df: pd.DataFrame, feature_col: str, target_col: str, task_type: str):
    """Analyze relationship between feature and target"""
    
    if task_type == "Classification":
        if df[feature_col].dtype in ['object', 'category']:
            # Categorical feature vs categorical target
            crosstab = pd.crosstab(df[feature_col], df[target_col])
            
            fig = px.imshow(
                crosstab.values,
                x=crosstab.columns,
                y=crosstab.index,
                title=f"{feature_col} vs {target_col}",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Numerical feature vs categorical target
            fig = px.box(
                df, 
                x=target_col, 
                y=feature_col,
                title=f"{feature_col} by {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        if df[feature_col].dtype in ['object', 'category']:
            # Categorical feature vs numerical target
            fig = px.box(
                df, 
                x=feature_col, 
                y=target_col,
                title=f"{target_col} by {feature_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Numerical feature vs numerical target
            fig = px.scatter(
                df, 
                x=feature_col, 
                y=target_col,
                title=f"{feature_col} vs {target_col}",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            correlation = df[feature_col].corr(df[target_col])
            st.metric("Correlation with Target", f"{correlation:.3f}")

if __name__ == "__main__":
    show()
