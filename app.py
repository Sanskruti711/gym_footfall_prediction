"""
🏋️ GYM OCCUPANCY PREDICTOR - MAIN APP
Streamlit dashboard for gym occupancy predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from our modules
try:
    data.database
    from models.predict import GymPredictor
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"❌ Module import error: {e}")
    st.info("Please run: `pip install -e .` or check your Python path")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="Gym Occupancy Predictor",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #93C5FD;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_database():
    """Initialize database connection"""
    try:
        db = GymDatabase('data/gym.db')
        if db.connect():
            return db
        else:
            st.error("❌ Could not connect to database")
            return None
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return None

@st.cache_resource
def init_predictor():
    """Initialize ML predictor"""
    try:
        predictor = GymPredictor()
        return predictor
    except Exception as e:
        st.error(f"❌ Predictor error: {e}")
        return None

def main():
    """Main application function"""
    
    # Title
    st.markdown('<h1 class="main-header">🏋️ University Gym Occupancy Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict optimal workout times to avoid crowds and maximize your gym experience!")
    
    # Initialize components
    db = init_database()
    predictor = init_predictor() if MODULES_LOADED else None
    
    # Sidebar for predictions
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔮 Make a Prediction</h2>', unsafe_allow_html=True)
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            hour = st.slider("Hour", 6, 22, 18, help="Gym hours: 6 AM to 10 PM")
            temperature = st.slider("Temperature (°C)", 10, 40, 25)
            
            day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_of_week = st.selectbox("Day", day_options)
            day_map = {day: idx for idx, day in enumerate(day_options)}
            day_value = day_map[day_of_week]
            
            is_weekend = 1 if day_value >= 5 else 0
        
        with col2:
            exam_period = st.radio("Exam Period?", ["No", "Yes"], horizontal=True)
            exam_value = 1 if exam_period == "Yes" else 0
            
            special_event = st.radio("Special Event?", ["No", "Yes"], horizontal=True)
            special_value = 1 if special_event == "Yes" else 0
            
            holiday = st.radio("Holiday?", ["No", "Yes"], horizontal=True)
            holiday_value = 1 if holiday == "Yes" else 0
            
            sports_event = st.radio("Sports Event?", ["No", "Yes"], horizontal=True)
            sports_value = 1 if sports_event == "Yes" else 0
            
            new_term_start = st.radio("New Term Start?", ["No", "Yes"], horizontal=True)
            new_term_value = 1 if new_term_start == "Yes" else 0
        
        # Previous day occupancy
        prev_occupancy = st.slider("Previous Day Occupancy (%)", 0, 100, 50,
                                 help="Occupancy at same hour yesterday")
        
        # Prediction button
        predict_button = st.button("🚀 Predict Occupancy", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown('<h3 class="sub-header">⚡ Quick Actions</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Data", use_container_width=True):
                st.session_state.show_data = True
        with col2:
            if st.button("📈 View Models", use_container_width=True):
                st.session_state.show_models = True
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "📊 Analytics", "🤖 Models", "⚙️ System"])
    
    with tab1:
        # Handle prediction if button was clicked
        if predict_button and predictor and db:
            # Prepare features
            features = {
                'hour': hour,
                'day_of_week': day_value,
                'exam_period': exam_value,
                'temperature': temperature,
                'is_weekend': is_weekend,
                'special_event': special_value,
                'holiday': holiday_value,
                'sports_event': sports_value,
                'new_term_start': new_term_value,
                'previous_day_occupancy': prev_occupancy
            }
            
            # Make prediction
            result = predictor.predict(features)
            
            # Save to database
            if db:
                db.save_prediction(features, result['average'], "v1.0")
            
            # Display results
            display_prediction_result(result, features)
        
        # Show heatmap
        display_heatmap(db)
        
        # Show recommendations
        display_recommendations(db)
    
    with tab2:
        display_analytics(db)
    
    with tab3:
        display_model_info(predictor)
    
    with tab4:
        display_system_info(db)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏋️ Gym Occupancy Prediction System • University Analytics Platform • v1.0</p>
        <p>📅 Hackathon 3 Submission • 👨‍💻 [Your Name] - [Your Reg No]</p>
    </div>
    """, unsafe_allow_html=True)

def display_prediction_result(result, features):
    """Display prediction results"""
    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Linear Regression", f"{result['linear_regression']}%")
    with col2:
        st.metric("Random Forest", f"{result['random_forest']}%")
    with col3:
        st.metric("Average", f"{result['average']}%")
    with col4:
        # Color-coded crowd level
        if result['color'] == 'green':
            st.success(f"👥 {result['crowd_level']}")
        elif result['color'] == 'orange':
            st.warning(f"👥 {result['crowd_level']}")
        else:
            st.error(f"👥 {result['crowd_level']}")
    
    # Recommendation card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown(f"### 💡 Recommendation")
    st.markdown(f"**{result['recommendation']}**")
    
    # Show features used
    with st.expander("📋 Features Used"):
        features_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
        st.dataframe(features_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_heatmap(db):
    """Display occupancy heatmap"""
    st.markdown('<h2 class="sub-header">🔥 Weekly Occupancy Heatmap</h2>', unsafe_allow_html=True)
    
    if db:
        # Get weekly summary from database
        weekly_data = db.get_weekly_summary()
        
        if not weekly_data.empty:
            # Create pivot table for heatmap
            heatmap_data = weekly_data.pivot(
                index='day_of_week', 
                columns='hour', 
                values='avg_occupancy'
            )
            
            # Convert day numbers to names
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            heatmap_data.index = [day_names[i] for i in heatmap_data.index]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn_r',
                zmin=0,
                zmax=100,
                text=heatmap_data.values,
                texttemplate='%{text:.0f}%',
                textfont={"size": 10},
                hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Occupancy: %{z:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="Average Occupancy by Day and Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available. Please generate data first.")
    else:
        st.warning("Database not connected")

def display_recommendations(db):
    """Display best time recommendations"""
    st.markdown('<h2 class="sub-header">🎯 Best Time Recommendations</h2>', unsafe_allow_html=True)
    
    if db:
        weekly_data = db.get_weekly_summary()
        if not weekly_data.empty:
            # Find best times (lowest occupancy)
            best_times = weekly_data.nsmallest(3, 'avg_occupancy')
            
            cols = st.columns(3)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            for idx, (_, row) in enumerate(best_times.iterrows()):
                day_name = day_names[int(row['day_of_week'])]
                hour_time = f"{int(row['hour'])}:00"
                occupancy = row['avg_occupancy']
                
                with cols[idx]:
                    st.metric(
                        label=f"{day_name}, {hour_time}",
                        value=f"{occupancy:.0f}%",
                        delta="Low crowd"
                    )
    else:
        st.warning("Database not connected")

def display_analytics(db):
    """Display analytics tab"""
    st.markdown('<h2 class="sub-header">📊 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if db:
        # Get daily statistics
        daily_stats = db.get_daily_stats()
        
        if not daily_stats.empty:
            # Line chart
            fig = px.line(daily_stats, x='date', y='daily_avg',
                         title="30-Day Occupancy Trend",
                         labels={'daily_avg': 'Average Occupancy (%)'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average", f"{daily_stats['daily_avg'].mean():.1f}%")
            with col2:
                st.metric("Maximum", f"{daily_stats['daily_max'].max():.0f}%")
            with col3:
                st.metric("Minimum", f"{daily_stats['daily_min'].min():.0f}%")
    else:
        st.warning("Database not connected")

def display_model_info(predictor):
    """Display model information"""
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    if predictor and predictor.models_loaded:
        # Feature importance
        importance_df = predictor.get_feature_importance()
        if importance_df is not None:
            fig = px.bar(importance_df, x='Feature', y='Importance',
                        title="Feature Importance (Random Forest)",
                        color='Importance')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'Best For': ['Linear time patterns', 'Complex interactions'],
            'Interpretability': ['High', 'Medium'],
            'Speed': ['Fast', 'Slower']
        })
        st.dataframe(comparison_data, use_container_width=True)
    else:
        st.warning("Models not loaded. Please train models first.")

def display_system_info(db):
    """Display system information"""
    st.markdown('<h2 class="sub-header">⚙️ System Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Database Stats")
        if db:
            try:
                occupancy_count = pd.read_sql_query("SELECT COUNT(*) FROM occupancy_data", db.conn).iloc[0,0]
                st.metric("Records", f"{occupancy_count:,}")
            except:
                st.write("No data")
    
    with col2:
        st.markdown("### 🔧 System Specs")
        st.write("**Python**: 3.9+")
        st.write("**ML Library**: scikit-learn")
        st.write("**Dashboard**: Streamlit")
        st.write("**Database**: SQLite")

if __name__ == "__main__":
    main()
