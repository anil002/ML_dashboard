"""
Advanced Data Upload Module
Support for multiple data sources with animations and interactive features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
import io
from typing import Dict, Any, Optional
import sqlite3
from sqlalchemy import create_engine
import json

# Import visualization helpers
from utils.helpers import display_dataframe_info, show_learning_tip

def show():
    """Advanced data upload interface with multiple sources"""
    
    st.title("🚀 Advanced Data Upload")
    st.markdown("Upload data from multiple sources with real-time preview and validation")
    
    # Add animated progress indicator
    if 'upload_progress' not in st.session_state:
        st.session_state.upload_progress = 0
    
    # Data source selection with icons
    st.markdown("## 📊 Choose Your Data Source")
    
    # Create tabs for different data sources
    source_tabs = st.tabs([
        "📁 File Upload", 
        "🌐 Web APIs", 
        "🗄️ Databases", 
        "📋 Manual Entry",
        "🔗 URL Import",
        "📊 Sample Datasets",
        "🔄 Real-time Data"
    ])
    
    with source_tabs[0]:
        handle_file_upload()
    
    with source_tabs[1]:
        handle_api_data()
    
    with source_tabs[2]:
        handle_database_connection()
    
    with source_tabs[3]:
        handle_manual_entry()
    
    with source_tabs[4]:
        handle_url_import()
    
    with source_tabs[5]:
        handle_sample_datasets()
    
    with source_tabs[6]:
        handle_realtime_data()

def handle_file_upload():
    """Enhanced file upload with drag-and-drop and preview"""
    
    st.markdown("### 📁 File Upload")
    st.markdown("Drag and drop files or browse to upload")
    
    # File upload with multiple formats
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv', 'txt'],
        accept_multiple_files=True,
        help="Supports CSV, Excel, JSON, Parquet, TSV, and TXT files"
    )
    
    if uploaded_files:
        # File selection if multiple files
        if len(uploaded_files) > 1:
            selected_file = st.selectbox(
                "Select file to process:",
                [f.name for f in uploaded_files]
            )
            uploaded_file = next(f for f in uploaded_files if f.name == selected_file)
        else:
            uploaded_file = uploaded_files[0]
        
        # Show file information with animation
        show_file_info_animated(uploaded_file)
        
        # Load data based on file type
        try:
            df = load_file_with_progress(uploaded_file)
            if df is not None:
                show_data_preview_interactive(df, f"📁 {uploaded_file.name}")
                
                if st.button("✅ Use This Dataset", type="primary"):
                    st.session_state.data = df
                    st.session_state.data_source = f"File: {uploaded_file.name}"
                    show_success_animation("Data loaded successfully!")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def handle_api_data():
    """Handle API data sources"""
    
    st.markdown("### 🌐 Web APIs")
    st.markdown("Connect to web APIs and fetch data")
    
    # Popular API presets
    api_presets = {
        "Custom API": {"url": "", "headers": {}},
        "JSONPlaceholder (Sample)": {
            "url": "https://jsonplaceholder.typicode.com/posts",
            "headers": {}
        },
        "COVID-19 Data": {
            "url": "https://disease.sh/v3/covid-19/countries",
            "headers": {}
        },
        "Public Weather API": {
            "url": "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m,relative_humidity_2m",
            "headers": {}
        }
    }
    
    selected_preset = st.selectbox("Choose API preset:", list(api_presets.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input(
            "API URL:",
            value=api_presets[selected_preset]["url"],
            placeholder="https://api.example.com/data"
        )
    
    with col2:
        api_key = st.text_input(
            "API Key (if required):",
            type="password",
            placeholder="Enter API key"
        )
    
    # Headers configuration
    with st.expander("🔧 Advanced Headers Configuration"):
        headers_json = st.text_area(
            "Custom Headers (JSON format):",
            value='{"Content-Type": "application/json"}',
            help="Add custom headers in JSON format"
        )
    
    if st.button("🚀 Fetch Data from API"):
        if api_url:
            fetch_api_data_with_animation(api_url, api_key, headers_json)

def handle_database_connection():
    """Handle database connections"""
    
    st.markdown("### 🗄️ Database Connection")
    st.markdown("Connect to various databases")
    
    db_type = st.selectbox(
        "Database Type:",
        ["SQLite", "PostgreSQL", "MySQL", "SQL Server", "MongoDB"]
    )
    
    if db_type == "SQLite":
        handle_sqlite_connection()
    else:
        handle_sql_connection(db_type)

def handle_sqlite_connection():
    """Handle SQLite database connection"""
    
    sqlite_file = st.file_uploader(
        "Upload SQLite database file:",
        type=['db', 'sqlite', 'sqlite3']
    )
    
    if sqlite_file:
        # Save uploaded file temporarily
        with open("temp_db.sqlite", "wb") as f:
            f.write(sqlite_file.getvalue())
        
        try:
            conn = sqlite3.connect("temp_db.sqlite")
            
            # Get table list
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table';", 
                conn
            )['name'].tolist()
            
            if tables:
                selected_table = st.selectbox("Select table:", tables)
                
                # Preview table
                preview_query = f"SELECT * FROM {selected_table} LIMIT 5"
                preview_df = pd.read_sql_query(preview_query, conn)
                
                st.markdown("#### 👀 Table Preview")
                st.dataframe(preview_df)
                
                # Custom query option
                with st.expander("🔍 Custom SQL Query"):
                    custom_query = st.text_area(
                        "SQL Query:",
                        value=f"SELECT * FROM {selected_table}",
                        height=100
                    )
                    
                    if st.button("Execute Query"):
                        try:
                            df = pd.read_sql_query(custom_query, conn)
                            show_data_preview_interactive(df, f"🗄️ SQLite: {selected_table}")
                            
                            if st.button("✅ Use Query Result", type="primary"):
                                st.session_state.data = df
                                st.session_state.data_source = f"SQLite: {sqlite_file.name}"
                                show_success_animation("Data loaded from database!")
                        
                        except Exception as e:
                            st.error(f"Query error: {str(e)}")
            
            conn.close()
        
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")

def handle_sql_connection(db_type: str):
    """Handle SQL database connections"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input("Host:", value="localhost")
        database = st.text_input("Database Name:")
    
    with col2:
        port = st.number_input("Port:", value=5432 if db_type == "PostgreSQL" else 3306)
        username = st.text_input("Username:")
    
    password = st.text_input("Password:", type="password")
    
    if st.button("🔗 Test Connection"):
        test_db_connection(db_type, host, port, database, username, password)

def handle_manual_entry():
    """Handle manual data entry"""
    
    st.markdown("### 📋 Manual Data Entry")
    st.markdown("Create datasets by entering data manually")
    
    entry_method = st.radio(
        "Entry Method:",
        ["Grid Editor", "CSV Text", "JSON Input"]
    )
    
    if entry_method == "Grid Editor":
        show_grid_editor()
    elif entry_method == "CSV Text":
        show_csv_text_editor()
    else:
        show_json_editor()

def handle_url_import():
    """Handle URL data import"""
    
    st.markdown("### 🔗 URL Data Import")
    st.markdown("Import data directly from URLs")
    
    url = st.text_input(
        "Data URL:",
        placeholder="https://example.com/data.csv"
    )
    
    url_type = st.selectbox(
        "URL Type:",
        ["CSV", "JSON", "Excel", "Auto-detect"]
    )
    
    if url and st.button("📥 Import from URL"):
        import_from_url_with_animation(url, url_type)

def handle_sample_datasets():
    """Enhanced sample datasets with categories"""
    
    st.markdown("### 📊 Sample Datasets")
    st.markdown("Explore with curated datasets for different use cases")
    
    # Categorize datasets
    dataset_categories = {
        "🏠 Real Estate": {
            "Boston Housing": "House prices with features",
            "California Housing": "Housing data from California census"
        },
        "💰 Finance": {
            "Stock Prices": "Historical stock market data",
            "Credit Approval": "Credit card approval decisions"
        },
        "🏥 Healthcare": {
            "Heart Disease": "Heart disease prediction data",
            "Diabetes": "Diabetes prediction dataset"
        },
        "🌸 Classification": {
            "Iris Flowers": "Classic flower classification",
            "Wine Quality": "Wine quality assessment",
            "Breast Cancer": "Cancer diagnosis data"
        },
        "🛒 Business": {
            "Customer Segmentation": "E-commerce customer data",
            "Sales Forecasting": "Retail sales data"
        }
    }
    
    # Category selection
    selected_category = st.selectbox(
        "Choose Category:",
        list(dataset_categories.keys())
    )
    
    # Dataset selection within category
    datasets_in_category = dataset_categories[selected_category]
    selected_dataset = st.selectbox(
        "Choose Dataset:",
        list(datasets_in_category.keys())
    )
    
    # Show dataset description
    st.info(f"📖 **{selected_dataset}**: {datasets_in_category[selected_dataset]}")
    
    if st.button(f"🚀 Load {selected_dataset}", type="primary"):
        load_sample_dataset_with_animation(selected_dataset)

def handle_realtime_data():
    """Handle real-time data streams"""
    
    st.markdown("### 🔄 Real-time Data")
    st.markdown("Connect to live data streams")
    
    realtime_source = st.selectbox(
        "Real-time Source:",
        ["Simulated Sensor Data", "Stock Market Feed", "Weather Updates", "Social Media Trends"]
    )
    
    if st.button("🔴 Start Real-time Stream"):
        start_realtime_stream(realtime_source)

# Helper functions for animations and interactivity

def show_file_info_animated(uploaded_file):
    """Show file information with animation"""
    
    # Create animated file info display
    info_container = st.container()
    
    with info_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Animated file size
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
            st.metric("📏 File Size", f"{file_size:.2f} MB")
        
        with col2:
            # File type with icon
            file_ext = uploaded_file.name.split('.')[-1].upper()
            st.metric("📄 File Type", file_ext)
        
        with col3:
            # Upload status with progress
            st.metric("✅ Status", "Ready")

def load_file_with_progress(uploaded_file):
    """Load file with animated progress bar"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Simulate loading steps with progress
        status_text.text("🔍 Analyzing file...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("📖 Reading data...")
        progress_bar.progress(50)
        
        # Load based on file type
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
        elif file_ext == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\t')
        
        progress_bar.progress(75)
        status_text.text("✅ Validating data...")
        time.sleep(0.5)
        
        progress_bar.progress(100)
        status_text.text("🎉 Data loaded successfully!")
        time.sleep(1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return df
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e

def show_data_preview_interactive(df: pd.DataFrame, title: str):
    """Show interactive data preview with animations"""
    
    st.markdown(f"### {title}")
    
    # Animated statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        st.metric("❓ Missing", f"{df.isnull().sum().sum()}")
    
    # Interactive tabs for data exploration
    preview_tabs = st.tabs(["📋 Data Preview", "📊 Statistics", "🔍 Data Types", "📈 Quick Viz"])
    
    with preview_tabs[0]:
        # Interactive data preview with filtering
        st.markdown("#### 👀 Data Preview")
        
        # Row selection
        num_rows = st.slider("Rows to display:", 5, min(100, len(df)), 10)
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns:",
            df.columns.tolist(),
            default=df.columns.tolist()[:5] if len(df.columns) > 5 else df.columns.tolist()
        )
        
        if selected_cols:
            st.dataframe(df[selected_cols].head(num_rows), use_container_width=True)
    
    with preview_tabs[1]:
        # Animated statistics
        st.markdown("#### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with preview_tabs[2]:
        # Data types with recommendations
        st.markdown("#### 🔍 Data Types Analysis")
        
        dtype_info = []
        for col in df.columns:
            dtype_info.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": f"{df[col].count():,}",
                "Null": f"{df[col].isnull().sum():,}",
                "Unique": f"{df[col].nunique():,}",
                "Sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        
        dtype_df = pd.DataFrame(dtype_info)
        st.dataframe(dtype_df, use_container_width=True)
    
    with preview_tabs[3]:
        # Quick visualizations
        st.markdown("#### 📈 Quick Visualizations")
        
        # Select numerical columns for quick plots
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            viz_col = st.selectbox("Select column for visualization:", numerical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=viz_col, title=f"Distribution of {viz_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=viz_col, title=f"Box Plot of {viz_col}")
                st.plotly_chart(fig, use_container_width=True)

def show_success_animation(message: str):
    """Show animated success message"""
    
    # Create success animation with balloons
    st.success(message)
    st.balloons()
    time.sleep(2)

def fetch_api_data_with_animation(url: str, api_key: str, headers_json: str):
    """Fetch API data with progress animation"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Parse headers
        headers = json.loads(headers_json) if headers_json else {}
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
        
        status_text.text("🌐 Connecting to API...")
        progress_bar.progress(25)
        
        response = requests.get(url, headers=headers, timeout=30)
        progress_bar.progress(50)
        
        status_text.text("📥 Downloading data...")
        response.raise_for_status()
        
        progress_bar.progress(75)
        status_text.text("🔄 Processing response...")
        
        # Try to parse as JSON
        data = response.json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
        else:
            st.error("Unable to convert API response to DataFrame")
            return
        
        progress_bar.progress(100)
        status_text.text("✅ API data loaded successfully!")
        
        # Clear progress
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Show data preview
        show_data_preview_interactive(df, f"🌐 API Data from {url}")
        
        if st.button("✅ Use API Data", type="primary"):
            st.session_state.data = df
            st.session_state.data_source = f"API: {url}"
            show_success_animation("API data loaded successfully!")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"API Error: {str(e)}")

def show_grid_editor():
    """Show interactive grid editor for manual data entry"""
    
    st.markdown("#### 📝 Grid Data Editor")
    
    # Initialize empty dataframe structure
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = pd.DataFrame({
            'Column1': ['', '', ''],
            'Column2': ['', '', ''],
            'Column3': ['', '', '']
        })
    
    # Column configuration
    col1, col2 = st.columns(2)
    
    with col1:
        num_cols = st.number_input("Number of columns:", 1, 20, 3)
    with col2:
        num_rows = st.number_input("Number of rows:", 1, 100, 5)
    
    # Generate column names
    col_names = [st.text_input(f"Column {i+1} name:", f"Column{i+1}") for i in range(num_cols)]
    
    # Create data editor
    if st.button("🔄 Update Grid"):
        st.session_state.manual_data = pd.DataFrame(
            {name: [''] * num_rows for name in col_names}
        )
    
    # Data editor
    edited_df = st.data_editor(
        st.session_state.manual_data,
        use_container_width=True,
        num_rows="dynamic"
    )
    
    if st.button("✅ Use Manual Data", type="primary"):
        # Remove empty rows
        clean_df = edited_df.dropna(how='all').reset_index(drop=True)
        
        if not clean_df.empty:
            st.session_state.data = clean_df
            st.session_state.data_source = "Manual Entry"
            show_success_animation("Manual data created successfully!")
        else:
            st.warning("Please enter some data first!")

def show_csv_text_editor():
    """Show CSV text editor"""
    
    st.markdown("#### 📝 CSV Text Editor")
    
    csv_text = st.text_area(
        "Enter CSV data:",
        value="Name,Age,City\nJohn,25,New York\nJane,30,Los Angeles\nBob,35,Chicago",
        height=200,
        help="Enter data in CSV format with headers"
    )
    
    if csv_text and st.button("📊 Parse CSV", type="primary"):
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            show_data_preview_interactive(df, "📝 CSV Text Data")
            
            if st.button("✅ Use CSV Data", type="primary"):
                st.session_state.data = df
                st.session_state.data_source = "CSV Text Entry"
                show_success_animation("CSV data loaded successfully!")
        
        except Exception as e:
            st.error(f"CSV parsing error: {str(e)}")

def show_json_editor():
    """Show JSON editor for data entry"""
    
    st.markdown("#### 🔧 JSON Data Editor")
    
    json_text = st.text_area(
        "Enter JSON data:",
        value='[{"name": "John", "age": 25, "city": "New York"}, {"name": "Jane", "age": 30, "city": "Los Angeles"}]',
        height=200,
        help="Enter data in JSON format (array of objects)"
    )
    
    if json_text and st.button("🔍 Parse JSON", type="primary"):
        try:
            data = json.loads(json_text)
            df = pd.json_normalize(data)
            show_data_preview_interactive(df, "🔧 JSON Data")
            
            if st.button("✅ Use JSON Data", type="primary"):
                st.session_state.data = df
                st.session_state.data_source = "JSON Entry"
                show_success_animation("JSON data loaded successfully!")
        
        except Exception as e:
            st.error(f"JSON parsing error: {str(e)}")

def import_from_url_with_animation(url: str, url_type: str):
    """Import data from URL with animation"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🌐 Fetching data from URL...")
        progress_bar.progress(33)
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        progress_bar.progress(66)
        status_text.text("📊 Processing data...")
        
        # Determine file type and parse
        if url_type == "Auto-detect":
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                url_type = "JSON"
            elif 'csv' in content_type or url.endswith('.csv'):
                url_type = "CSV"
            elif url.endswith(('.xlsx', '.xls')):
                url_type = "Excel"
            else:
                url_type = "CSV"  # Default
        
        if url_type == "CSV":
            df = pd.read_csv(io.StringIO(response.text))
        elif url_type == "JSON":
            data = response.json()
            df = pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.DataFrame(data)
        elif url_type == "Excel":
            df = pd.read_excel(io.BytesIO(response.content))
        
        progress_bar.progress(100)
        status_text.text("✅ URL data loaded successfully!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        show_data_preview_interactive(df, f"🔗 URL Data: {url}")
        
        if st.button("✅ Use URL Data", type="primary"):
            st.session_state.data = df
            st.session_state.data_source = f"URL: {url}"
            show_success_animation("URL data imported successfully!")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"URL import error: {str(e)}")

def load_sample_dataset_with_animation(dataset_name: str):
    """Load sample dataset with animation"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"📦 Loading {dataset_name}...")
        progress_bar.progress(33)
        
        # Load different sample datasets
        if dataset_name == "Iris Flowers":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            df = df.drop('target', axis=1)
        
        elif dataset_name == "Boston Housing":
            # Create synthetic Boston-like housing data
            np.random.seed(42)
            n_samples = 506
            df = pd.DataFrame({
                'CRIM': np.random.exponential(2, n_samples),
                'ZN': np.random.choice([0, 12.5, 25], n_samples),
                'INDUS': np.random.uniform(0, 30, n_samples),
                'RM': np.random.normal(6.3, 0.7, n_samples),
                'AGE': np.random.uniform(0, 100, n_samples),
                'DIS': np.random.exponential(3, n_samples),
                'MEDV': np.random.normal(22, 8, n_samples)
            })
        
        elif dataset_name == "Wine Quality":
            # Generate synthetic wine data
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'fixed_acidity': np.random.normal(7.2, 1.3, n_samples),
                'volatile_acidity': np.random.normal(0.34, 0.18, n_samples),
                'citric_acid': np.random.normal(0.32, 0.15, n_samples),
                'residual_sugar': np.random.exponential(3, n_samples),
                'chlorides': np.random.normal(0.05, 0.02, n_samples),
                'pH': np.random.normal(3.2, 0.15, n_samples),
                'alcohol': np.random.normal(10.5, 1.2, n_samples),
                'quality': np.random.choice(range(3, 10), n_samples)
            })
        
        else:
            # Generate generic sample data
            np.random.seed(42)
            df = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 1000),
                'feature_2': np.random.normal(5, 2, 1000),
                'feature_3': np.random.exponential(1, 1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.choice([0, 1], 1000)
            })
        
        progress_bar.progress(66)
        status_text.text("🔍 Validating dataset...")
        time.sleep(0.5)
        
        progress_bar.progress(100)
        status_text.text("✅ Sample dataset loaded!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        show_data_preview_interactive(df, f"📊 {dataset_name}")
        
        if st.button("✅ Use Sample Dataset", type="primary"):
            st.session_state.data = df
            st.session_state.data_source = f"Sample: {dataset_name}"
            show_success_animation(f"{dataset_name} loaded successfully!")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error loading sample dataset: {str(e)}")

def start_realtime_stream(source_type: str):
    """Start real-time data stream simulation"""
    
    st.markdown(f"### 🔴 Live: {source_type}")
    
    # Create containers for real-time updates
    metrics_container = st.container()
    chart_container = st.container()
    
    # Initialize data storage
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = []
    
    # Simulate real-time data
    for i in range(10):  # 10 updates
        # Generate data based on source type
        if source_type == "Simulated Sensor Data":
            data_point = {
                'timestamp': pd.Timestamp.now() + pd.Timedelta(seconds=i),
                'temperature': 20 + np.random.normal(0, 2),
                'humidity': 50 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 5)
            }
        
        elif source_type == "Stock Market Feed":
            data_point = {
                'timestamp': pd.Timestamp.now() + pd.Timedelta(seconds=i),
                'price': 100 + np.random.normal(0, 2),
                'volume': np.random.randint(1000, 10000),
                'symbol': 'SAMPLE'
            }
        
        else:
            data_point = {
                'timestamp': pd.Timestamp.now() + pd.Timedelta(seconds=i),
                'value': np.random.normal(0, 1),
                'category': np.random.choice(['A', 'B', 'C'])
            }
        
        st.session_state.realtime_data.append(data_point)
        
        # Update displays
        with metrics_container:
            cols = st.columns(len(data_point) - 1)  # Exclude timestamp
            for j, (key, value) in enumerate(list(data_point.items())[1:]):
                if isinstance(value, (int, float)):
                    cols[j].metric(key.title(), f"{value:.2f}")
                else:
                    cols[j].metric(key.title(), str(value))
        
        # Update chart
        if len(st.session_state.realtime_data) > 1:
            df = pd.DataFrame(st.session_state.realtime_data)
            
            with chart_container:
                if source_type == "Simulated Sensor Data":
                    fig = px.line(df, x='timestamp', y=['temperature', 'humidity'], 
                                 title="Real-time Sensor Data")
                elif source_type == "Stock Market Feed":
                    fig = px.line(df, x='timestamp', y='price', 
                                 title="Real-time Stock Price")
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig = px.line(df, x='timestamp', y=numeric_cols[0], 
                                     title="Real-time Data Stream")
                
                st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)  # Update every second
    
    # Option to save real-time data
    if st.button("💾 Save Real-time Data", type="primary"):
        df = pd.DataFrame(st.session_state.realtime_data)
        st.session_state.data = df
        st.session_state.data_source = f"Real-time: {source_type}"
        show_success_animation("Real-time data saved successfully!")

def test_db_connection(db_type: str, host: str, port: int, database: str, username: str, password: str):
    """Test database connection"""
    
    try:
        # Create connection string based on database type
        if db_type == "PostgreSQL":
            conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "MySQL":
            conn_str = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "SQL Server":
            conn_str = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        engine = create_engine(conn_str)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1").fetchone()
            st.success("✅ Database connection successful!")
            
            # Show available tables
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables = pd.read_sql(tables_query, engine)
            
            if not tables.empty:
                st.markdown("#### Available Tables:")
                st.dataframe(tables, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")

if __name__ == "__main__":
    show()
