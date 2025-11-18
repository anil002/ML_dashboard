"""
User Experience Enhancement Module
Personalized dashboard experience for different user types
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta

def show():
    """Enhanced user experience interface"""
    
    st.title("👤 Personalized ML Experience")
    st.markdown("Customize your machine learning journey based on your experience level and goals")
    
    # User profile setup
    setup_user_profile()
    
    # Show personalized dashboard
    if 'user_profile' in st.session_state and st.session_state.user_profile:
        # Check if profile is complete
        profile = st.session_state.user_profile
        required_keys = ['experience_level', 'primary_goal', 'learning_style', 'domain_focus', 'time_availability']
        
        if all(key in profile for key in required_keys):
            show_personalized_dashboard()
        else:
            st.info("👋 Please complete your profile setup above to get a personalized experience!")
    else:
        st.info("👋 Welcome! Please set up your profile above to get started with a personalized ML experience.")

def setup_user_profile():
    """Setup user profile and preferences"""
    
    # Initialize empty profile if not exists
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    
    profile_complete = False
    if st.session_state.user_profile:
        required_keys = ['experience_level', 'primary_goal', 'learning_style', 'domain_focus', 'time_availability']
        profile_complete = all(key in st.session_state.user_profile for key in required_keys)
    
    # Show setup form
    expand_setup = not profile_complete
    
    with st.expander("🎯 Setup Your Profile", expanded=expand_setup):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Experience level
            experience_level = st.selectbox(
                "What's your ML experience level?",
                ["🌱 Beginner", "🌿 Intermediate", "🌳 Advanced", "🚀 Expert"],
                help="This will customize the interface complexity and explanations"
            )
            
            # Primary goal
            primary_goal = st.selectbox(
                "What's your primary goal?",
                ["📚 Learning ML Concepts", "🔬 Research & Experimentation", "🏢 Business Analytics", "🎓 Academic Project", "🚀 Production Deployment"],
                help="This will customize recommendations and workflows"
            )
            
            # Preferred learning style
            learning_style = st.selectbox(
                "How do you prefer to learn?",
                ["👀 Visual/Interactive", "📖 Reading/Documentation", "🎬 Step-by-step Tutorials", "🛠️ Hands-on Practice"],
                help="This will customize how information is presented"
            )
        
        with col2:
            # Domain focus
            domain_focus = st.multiselect(
                "Which domains interest you most?",
                ["🏥 Healthcare", "💰 Finance", "🛒 E-commerce", "🎮 Gaming", "🌐 Social Media", "🏭 Manufacturing", "🚗 Transportation", "🔬 Research", "📊 General Analytics"],
                help="This will customize sample datasets and use cases"
            )
            
            # Time availability
            time_availability = st.selectbox(
                "How much time do you typically have?",
                ["⚡ Quick (15-30 min)", "⏰ Standard (30-60 min)", "🕐 Extended (1-2 hours)", "📅 Deep Dive (2+ hours)"],
                help="This will customize workflow suggestions"
            )
            
            # Notification preferences
            notifications = st.multiselect(
                "What notifications would you like?",
                ["💡 Learning Tips", "⚠️ Best Practice Warnings", "🎉 Achievement Unlocks", "📊 Progress Updates", "🔔 Reminders"],
                default=["💡 Learning Tips", "⚠️ Best Practice Warnings"]
            )
        
        if st.button("💾 Save Profile", type="primary"):
            st.session_state.user_profile = {
                'experience_level': experience_level,
                'primary_goal': primary_goal,
                'learning_style': learning_style,
                'domain_focus': domain_focus,
                'time_availability': time_availability,
                'notifications': notifications,
                'created_at': datetime.now().isoformat(),
                'progress': initialize_progress()
            }
            st.success("✅ Profile saved! Your dashboard is now personalized.")
            st.rerun()

def initialize_progress():
    """Initialize user progress tracking"""
    return {
        'modules_completed': [],
        'datasets_explored': [],
        'models_trained': [],
        'achievements': [],
        'total_time_spent': 0,
        'skill_points': 0,
        'current_streak': 0,
        'last_activity': datetime.now().isoformat()
    }

def show_personalized_dashboard():
    """Show dashboard customized for user profile"""
    
    profile = st.session_state.user_profile
    
    # Welcome message based on profile
    show_personalized_welcome(profile)
    
    # Adaptive interface based on experience
    experience_level = profile.get('experience_level', '🌱 Beginner')
    
    if experience_level == "🌱 Beginner":
        show_beginner_interface(profile)
    elif experience_level == "🌿 Intermediate":
        show_intermediate_interface(profile)
    elif experience_level == "🌳 Advanced":
        show_advanced_interface(profile)
    else:  # Expert
        show_expert_interface(profile)

def show_personalized_welcome(profile: Dict):
    """Show personalized welcome message"""
    
    # Safely extract values with defaults
    experience = profile.get('experience_level', '🌱 Beginner').split(' ')[1] if profile.get('experience_level') else 'Beginner'
    goal = profile.get('primary_goal', '📚 Learning ML Concepts').split(' ', 1)[1] if profile.get('primary_goal') else 'Learning ML Concepts'
    
    # Progress tracking
    progress = profile.get('progress', {})
    
    st.markdown(f"## Welcome back, {experience} ML Practitioner! 👋")
    st.markdown(f"🎯 **Goal:** {goal}")
    
    # Progress overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Skill Points", progress.get('skill_points', 0))
    with col2:
        st.metric("📚 Modules Completed", len(progress.get('modules_completed', [])))
    with col3:
        st.metric("🔥 Current Streak", f"{progress.get('current_streak', 0)} days")
    with col4:
        st.metric("⏱️ Time Invested", f"{progress.get('total_time_spent', 0):.1f}h")
    
    # Achievement badges
    show_achievement_badges(progress.get('achievements', []))

def show_achievement_badges(achievements: List[str]):
    """Show achievement badges"""
    
    all_achievements = {
        "first_dataset": {"emoji": "📊", "title": "Data Explorer", "description": "Loaded your first dataset"},
        "first_model": {"emoji": "🧠", "title": "Model Builder", "description": "Trained your first model"},
        "perfect_score": {"emoji": "🎯", "title": "Accuracy Master", "description": "Achieved 95%+ accuracy"},
        "data_scientist": {"emoji": "🔬", "title": "Data Scientist", "description": "Completed full ML pipeline"},
        "speed_runner": {"emoji": "⚡", "title": "Speed Runner", "description": "Completed analysis in under 15 minutes"},
        "perfectionist": {"emoji": "💎", "title": "Perfectionist", "description": "Zero missing values in dataset"},
        "explorer": {"emoji": "🗺️", "title": "Algorithm Explorer", "description": "Tried 5+ different algorithms"},
        "teacher": {"emoji": "👨‍🏫", "title": "Teacher", "description": "Used explanations module extensively"}
    }
    
    if achievements:
        st.markdown("### 🏅 Your Achievements")
        
        # Display badges in a grid
        cols = st.columns(min(len(achievements), 4))
        
        for i, achievement_id in enumerate(achievements):
            if achievement_id in all_achievements:
                achievement = all_achievements[achievement_id]
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px solid gold; border-radius: 10px; margin: 5px;">
                        <div style="font-size: 2em;">{achievement['emoji']}</div>
                        <div style="font-weight: bold;">{achievement['title']}</div>
                        <div style="font-size: 0.8em; color: gray;">{achievement['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

def show_beginner_interface(profile: Dict):
    """Show interface optimized for beginners"""
    
    st.markdown("### 🌱 Beginner's ML Journey")
    
    # Learning path recommendation
    with st.container():
        st.markdown("#### 🗺️ Recommended Learning Path")
        
        steps = [
            {"step": 1, "title": "📊 Upload & Explore Data", "description": "Start with sample datasets to understand data structure", "completed": "data_upload" in profile.get('progress', {}).get('modules_completed', [])},
            {"step": 2, "title": "🔍 Data Analysis", "description": "Learn to analyze and visualize your data", "completed": "eda" in profile.get('progress', {}).get('modules_completed', [])},
            {"step": 3, "title": "🧹 Data Cleaning", "description": "Prepare your data for machine learning", "completed": "preprocessing" in profile.get('progress', {}).get('modules_completed', [])},
            {"step": 4, "title": "🧠 First Model", "description": "Train your first machine learning model", "completed": "model_training" in profile.get('progress', {}).get('modules_completed', [])},
            {"step": 5, "title": "📈 Understand Results", "description": "Learn to evaluate and interpret model performance", "completed": "evaluation" in profile.get('progress', {}).get('modules_completed', [])}
        ]
        
        for step in steps:
            status_icon = "✅" if step["completed"] else "⏳"
            completion_color = "success" if step["completed"] else "info"
            
            with st.container():
                if step["completed"]:
                    st.success(f"{status_icon} **Step {step['step']}: {step['title']}** - {step['description']}")
                else:
                    st.info(f"{status_icon} **Step {step['step']}: {step['title']}** - {step['description']}")
    
    # Quick start options
    st.markdown("#### 🚀 Quick Start Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌸 Try Iris Dataset", help="Perfect for beginners - simple flower classification"):
            load_sample_for_beginner("iris")
    
    with col2:
        if st.button("🏠 Try Housing Data", help="Learn regression with house price prediction"):
            load_sample_for_beginner("housing")
    
    with col3:
        if st.button("📚 View Tutorial", help="Step-by-step guided tutorial"):
            show_guided_tutorial()
    
    # Learning tips for beginners
    show_beginner_tips(profile)

def show_intermediate_interface(profile: Dict):
    """Show interface for intermediate users"""
    
    st.markdown("### 🌿 Intermediate ML Workspace")
    
    # Quick access to advanced features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛠️ Advanced Tools")
        
        if st.button("🔧 Hyperparameter Tuning"):
            st.info("Navigate to Model Training > Hyperparameter Tuning for advanced optimization")
        
        if st.button("📊 Custom Visualizations"):
            show_custom_visualization_builder()
        
        if st.button("🔍 Feature Engineering"):
            st.info("Explore advanced feature engineering in the Preprocessing module")
    
    with col2:
        st.markdown("#### 📈 Performance Optimization")
        
        # Show performance tips
        performance_tips = [
            "🚀 Use cross-validation for robust model evaluation",
            "⚖️ Consider ensemble methods for better accuracy",
            "🎯 Focus on feature selection to reduce overfitting",
            "📏 Scale features for distance-based algorithms"
        ]
        
        for tip in performance_tips:
            st.write(tip)
    
    # Project templates
    show_project_templates(profile)

def show_advanced_interface(profile: Dict):
    """Show interface for advanced users"""
    
    st.markdown("### 🌳 Advanced ML Laboratory")
    
    # Advanced analytics dashboard
    show_advanced_analytics_dashboard(profile)
    
    # Research tools
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔬 Research Tools")
        
        if st.button("📊 Statistical Analysis"):
            show_statistical_analysis_tools()
        
        if st.button("🧪 Experiment Tracking"):
            show_experiment_tracking()
        
        if st.button("🔄 A/B Testing"):
            show_ab_testing_tools()
    
    with col2:
        st.markdown("#### 🚀 Deployment Prep")
        
        if st.button("📦 Model Packaging"):
            show_model_packaging_tools()
        
        if st.button("⚡ Performance Profiling"):
            show_performance_profiling()
        
        if st.button("🔍 Model Interpretability"):
            show_advanced_interpretability()

def show_expert_interface(profile: Dict):
    """Show interface for expert users"""
    
    st.markdown("### 🚀 Expert ML Command Center")
    
    # Full control panel
    st.markdown("#### ⚡ Quick Actions")
    
    expert_actions = [
        ("🔧 Custom Algorithm Implementation", "implement_custom_algorithm"),
        ("📊 Multi-model Ensemble", "create_ensemble"),
        ("🎯 AutoML Pipeline", "setup_automl"),
        ("🔍 Advanced Feature Selection", "feature_selection"),
        ("⚡ Performance Optimization", "optimize_performance"),
        ("🚀 Production Deployment", "deploy_model")
    ]
    
    cols = st.columns(3)
    for i, (action, key) in enumerate(expert_actions):
        with cols[i % 3]:
            if st.button(action):
                handle_expert_action(key)
    
    # Advanced metrics and monitoring
    show_expert_metrics_dashboard(profile)

def load_sample_for_beginner(dataset_type: str):
    """Load sample dataset optimized for beginners"""
    
    if dataset_type == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['species'] = data.target_names[data.target]
        
        st.session_state.data = df
        st.session_state.data_source = "Sample: Iris (Beginner)"
        
        # Add to progress
        update_progress("datasets_explored", "iris")
        award_achievement("first_dataset")
        
        st.success("🌸 Iris dataset loaded! This dataset is perfect for learning classification.")
        st.info("💡 **Beginner Tip:** The Iris dataset has 4 features (measurements) and 3 species to predict. It's small and clean - perfect for learning!")
    
    elif dataset_type == "housing":
        # Create beginner-friendly housing dataset
        np.random.seed(42)
        n_samples = 200  # Smaller for beginners
        
        df = pd.DataFrame({
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'square_feet': np.random.randint(800, 3000, n_samples),
            'age_years': np.random.randint(0, 50, n_samples),
            'price': np.random.randint(100000, 800000, n_samples)
        })
        
        st.session_state.data = df
        st.session_state.data_source = "Sample: Housing (Beginner)"
        
        update_progress("datasets_explored", "housing")
        
        st.success("🏠 Housing dataset loaded! Perfect for learning regression (predicting prices).")
        st.info("💡 **Beginner Tip:** This dataset helps you predict house prices based on features like bedrooms, bathrooms, and size.")

def show_guided_tutorial():
    """Show step-by-step guided tutorial"""
    
    st.markdown("### 📚 Guided ML Tutorial")
    
    tutorial_steps = [
        {
            "title": "🎯 Understanding Machine Learning",
            "content": """
            **Machine Learning** is like teaching a computer to recognize patterns and make predictions.
            
            **Key Concepts:**
            - **Features**: The input data (like height, weight, age)
            - **Target**: What you want to predict (like disease, price, category)
            - **Training**: Teaching the computer using examples
            - **Prediction**: Using the trained model on new data
            """,
            "action": "Next: Load Data"
        },
        {
            "title": "📊 Loading Your First Dataset", 
            "content": """
            **Data** is the foundation of machine learning. Let's start with a simple dataset.
            
            **Steps:**
            1. Go to 'Data Upload' tab
            2. Click 'Load Iris Dataset'
            3. Observe the data structure
            
            **What to look for:**
            - Number of rows (samples)
            - Number of columns (features)
            - Data types (numbers, text)
            """,
            "action": "Next: Explore Data"
        }
        # Add more tutorial steps...
    ]
    
    # Tutorial progress
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 0
    
    current_step = tutorial_steps[st.session_state.tutorial_step]
    
    st.markdown(f"#### Step {st.session_state.tutorial_step + 1}: {current_step['title']}")
    st.markdown(current_step['content'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.tutorial_step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.tutorial_step -= 1
                st.rerun()
    
    with col2:
        if st.session_state.tutorial_step < len(tutorial_steps) - 1:
            if st.button(current_step['action']):
                st.session_state.tutorial_step += 1
                st.rerun()

def show_beginner_tips(profile: Dict):
    """Show helpful tips for beginners"""
    
    tips = [
        "💡 Start with clean, small datasets to understand the basics",
        "📊 Always visualize your data before building models",
        "🎯 Focus on understanding rather than achieving perfect accuracy",
        "🔄 Experiment with different algorithms to see how they behave",
        "📚 Read the explanations in the Algorithm Guide section"
    ]
    
    if "💡 Learning Tips" in profile.get('notifications', []):
        with st.expander("💡 Beginner Tips", expanded=True):
            for tip in tips:
                st.info(tip)

def show_project_templates(profile: Dict):
    """Show project templates for different goals"""
    
    st.markdown("#### 📋 Project Templates")
    
    templates = {
        "🏥 Healthcare Analysis": {
            "description": "Analyze patient data and predict health outcomes",
            "datasets": ["Heart Disease", "Diabetes"],
            "algorithms": ["Logistic Regression", "Random Forest"],
            "focus": "High precision and interpretability"
        },
        "💰 Financial Prediction": {
            "description": "Predict stock prices or credit risk",
            "datasets": ["Stock Data", "Credit Approval"],
            "algorithms": ["Time Series", "Gradient Boosting"],
            "focus": "Handling temporal data and risk assessment"
        },
        "🛒 Customer Analytics": {
            "description": "Understand customer behavior and preferences",
            "datasets": ["Purchase History", "Customer Segments"],
            "algorithms": ["Clustering", "Classification"],
            "focus": "Segmentation and recommendation systems"
        }
    }
    
    selected_template = st.selectbox("Choose a project template:", list(templates.keys()))
    
    if selected_template:
        template = templates[selected_template]
        
        with st.container():
            st.markdown(f"**{selected_template}**")
            st.write(template["description"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Recommended Datasets:**")
                for dataset in template["datasets"]:
                    st.write(f"• {dataset}")
            
            with col2:
                st.markdown("**Suggested Algorithms:**")
                for algorithm in template["algorithms"]:
                    st.write(f"• {algorithm}")
            
            with col3:
                st.markdown("**Key Focus:**")
                st.write(template["focus"])
            
            if st.button(f"🚀 Start {selected_template} Project"):
                start_template_project(selected_template, template)

def update_progress(category: str, item: str):
    """Update user progress"""
    
    if 'user_profile' in st.session_state and 'progress' in st.session_state.user_profile:
        progress = st.session_state.user_profile['progress']
        
        if category not in progress:
            progress[category] = []
        
        if item not in progress[category]:
            progress[category].append(item)
            progress['skill_points'] = progress.get('skill_points', 0) + 10
            progress['last_activity'] = datetime.now().isoformat()

def award_achievement(achievement_id: str):
    """Award achievement to user"""
    
    if 'user_profile' in st.session_state and 'progress' in st.session_state.user_profile:
        progress = st.session_state.user_profile['progress']
        achievements = progress.get('achievements', [])
        
        if achievement_id not in achievements:
            achievements.append(achievement_id)
            progress['achievements'] = achievements
            progress['skill_points'] = progress.get('skill_points', 0) + 50
            
            # Show achievement notification
            st.balloons()
            st.success(f"🏆 Achievement Unlocked: {achievement_id.replace('_', ' ').title()}!")

# Additional helper functions for advanced interfaces
def show_custom_visualization_builder():
    """Show custom visualization builder"""
    st.info("🔧 Custom Visualization Builder - Advanced feature for creating custom plots")

def show_advanced_analytics_dashboard(profile: Dict):
    """Show advanced analytics dashboard"""
    st.info("📊 Advanced Analytics Dashboard - Coming soon for power users")

def show_statistical_analysis_tools():
    """Show statistical analysis tools"""
    st.info("📊 Statistical Analysis Tools - Advanced statistical testing and analysis")

def show_experiment_tracking():
    """Show experiment tracking tools"""
    st.info("🧪 Experiment Tracking - Track and compare multiple experiments")

def show_ab_testing_tools():
    """Show A/B testing tools"""
    st.info("🔄 A/B Testing Tools - Set up and analyze A/B tests")

def show_model_packaging_tools():
    """Show model packaging tools"""
    st.info("📦 Model Packaging - Package models for deployment")

def show_performance_profiling():
    """Show performance profiling tools"""
    st.info("⚡ Performance Profiling - Analyze model performance bottlenecks")

def show_advanced_interpretability():
    """Show advanced interpretability tools"""
    st.info("🔍 Advanced Interpretability - Deep model analysis and explanation")

def show_expert_metrics_dashboard(profile: Dict):
    """Show expert metrics dashboard"""
    st.info("📊 Expert Metrics Dashboard - Comprehensive model monitoring")

def handle_expert_action(action: str):
    """Handle expert-level actions"""
    st.info(f"🚀 Expert Action: {action} - Advanced functionality")

def start_template_project(template_name: str, template: Dict):
    """Start a project from template"""
    st.success(f"🚀 Starting {template_name} project!")
    st.info("This will guide you through a complete project workflow tailored to your chosen domain.")

if __name__ == "__main__":
    show()
