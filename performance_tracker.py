# performance_tracker.py - Comprehensive Analytics and Performance Tracking

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import hashlib
import time
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable.*', category=UserWarning)

class PerformanceTracker:
    """
    Comprehensive performance tracking for Lumen Navigator
    Tracks queries, responses, user satisfaction, and system performance
    """
    
    def __init__(self, db_path: str = "lumen_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main query tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_id TEXT,
            timestamp DATETIME,
            query_text TEXT,
            query_length INTEGER,
            query_category TEXT,
            has_attachments BOOLEAN,
            attachment_count INTEGER,
            attachment_types TEXT,
            response_text TEXT,
            response_length INTEGER,
            response_time_seconds REAL,
            sources_count INTEGER,
            sources_list TEXT,
            performance_mode TEXT,
            rag_system_version TEXT,
            error_occurred BOOLEAN,
            error_message TEXT
        )
        ''')
        
        # User feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id TEXT PRIMARY KEY,
            query_id TEXT,
            session_id TEXT,
            user_id TEXT,
            timestamp DATETIME,
            rating INTEGER,
            feedback_type TEXT,
            feedback_text TEXT,
            usefulness_score INTEGER,
            accuracy_score INTEGER,
            clarity_score INTEGER,
            FOREIGN KEY (query_id) REFERENCES query_logs (id)
        )
        ''')
        
        # System performance metrics
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            metric_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            additional_data TEXT
        )
        ''')
        
        # User sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            start_timestamp DATETIME,
            end_timestamp DATETIME,
            total_queries INTEGER,
            avg_response_time REAL,
            user_satisfaction_avg REAL,
            device_info TEXT,
            browser_info TEXT
        )
        ''')
        
        # Feature usage tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_usage (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_id TEXT,
            timestamp DATETIME,
            feature_name TEXT,
            feature_action TEXT,
            additional_context TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        if 'analytics_session_id' not in st.session_state:
            st.session_state.analytics_session_id = str(uuid.uuid4())
        return st.session_state.analytics_session_id
    
    def get_user_id(self) -> str:
        """Get user ID from Auth0 or generate anonymous ID"""
        try:
            # Try to get user info from Auth0
            if 'user_info' in st.session_state and st.session_state.user_info:
                user_email = st.session_state.user_info.get('email', '')
                # Hash email for privacy
                return hashlib.sha256(user_email.encode()).hexdigest()[:16]
        except:
            pass
        
        # Fallback to anonymous session-based ID
        if 'analytics_user_id' not in st.session_state:
            st.session_state.analytics_user_id = f"anon_{str(uuid.uuid4())[:8]}"
        return st.session_state.analytics_user_id
    
    def classify_query(self, query_text: str) -> str:
        """Classify query into categories for analysis"""
        query_lower = query_text.lower()
        
        # Define keywords for different categories
        categories = {
            'ofsted': ['ofsted', 'inspection', 'outstanding', 'requires improvement', 'inadequate'],
            'safeguarding': ['safeguarding', 'protection', 'risk', 'safety', 'abuse', 'concern'],
            'compliance': ['compliance', 'regulation', 'policy', 'procedure', 'standard'],
            'training': ['training', 'development', 'course', 'skill', 'education'],
            'documentation': ['document', 'record', 'report', 'form', 'file'],
            'staffing': ['staff', 'employee', 'recruitment', 'supervision', 'management'],
            'young_people': ['young people', 'children', 'resident', 'placement', 'care plan'],
            'general': []  # Default category
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def log_query(self, 
                  query_text: str, 
                  response_text: str, 
                  response_time: float,
                  sources: List[Dict] = None,
                  attachments: List = None,
                  performance_mode: str = "balanced",
                  error_info: Dict = None) -> str:
        """Log a query and response"""
        
        query_id = str(uuid.uuid4())
        session_id = self.generate_session_id()
        user_id = self.get_user_id()
        timestamp = datetime.now()
        
        # Process attachments info
        has_attachments = bool(attachments)
        attachment_count = len(attachments) if attachments else 0
        attachment_types = json.dumps([f.type if hasattr(f, 'type') else 'unknown' 
                                     for f in attachments]) if attachments else None
        
        # Process sources
        sources_count = len(sources) if sources else 0
        sources_list = json.dumps([s.get('title', 'Unknown') for s in sources]) if sources else None
        
        # Classify query
        query_category = self.classify_query(query_text)
        
        # Error handling
        error_occurred = error_info is not None
        error_message = json.dumps(error_info) if error_info else None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO query_logs 
        (id, session_id, user_id, timestamp, query_text, query_length, query_category,
         has_attachments, attachment_count, attachment_types, response_text, response_length,
         response_time_seconds, sources_count, sources_list, performance_mode, 
         rag_system_version, error_occurred, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            query_id, session_id, user_id, timestamp, query_text, len(query_text),
            query_category, has_attachments, attachment_count, attachment_types,
            response_text, len(response_text), response_time, sources_count, sources_list,
            performance_mode, "v1.0", error_occurred, error_message
        ))
        
        conn.commit()
        conn.close()
        
        # Store query_id in session for feedback collection
        st.session_state.current_query_id = query_id
        
        return query_id
    
    def log_user_feedback(self, 
                         query_id: str, 
                         rating: int, 
                         feedback_type: str = "rating",
                         feedback_text: str = None,
                         usefulness_score: int = None,
                         accuracy_score: int = None,
                         clarity_score: int = None):
        """Log user feedback for a query"""
        
        feedback_id = str(uuid.uuid4())
        session_id = self.generate_session_id()
        user_id = self.get_user_id()
        timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_feedback 
        (id, query_id, session_id, user_id, timestamp, rating, feedback_type,
         feedback_text, usefulness_score, accuracy_score, clarity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id, query_id, session_id, user_id, timestamp, rating,
            feedback_type, feedback_text, usefulness_score, accuracy_score, clarity_score
        ))
        
        conn.commit()
        conn.close()
    
    def log_feature_usage(self, feature_name: str, action: str, context: Dict = None):
        """Log feature usage for UX analysis"""
        
        feature_id = str(uuid.uuid4())
        session_id = self.generate_session_id()
        user_id = self.get_user_id()
        timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO feature_usage 
        (id, session_id, user_id, timestamp, feature_name, feature_action, additional_context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature_id, session_id, user_id, timestamp, feature_name, action,
            json.dumps(context) if context else None
        ))
        
        conn.commit()
        conn.close()
    
    def log_system_metric(self, metric_type: str, metric_name: str, value: float, data: Dict = None):
        """Log system performance metrics"""
        
        metric_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO system_metrics 
        (id, timestamp, metric_type, metric_name, metric_value, additional_data)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric_id, timestamp, metric_type, metric_name, value,
            json.dumps(data) if data else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_dashboard_data(self, days: int = 30) -> Dict:
        """Get comprehensive analytics data for dashboard"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create SQLAlchemy engine to eliminate pandas warnings
        engine = create_engine(f'sqlite:///{self.db_path}')
        
        # Query metrics - FIXED: Using SQLAlchemy engine instead of direct connection
        query_df = pd.read_sql_query('''
        SELECT * FROM query_logs 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp DESC
        ''', engine, params=(start_date, end_date))
        
        # Feedback metrics - FIXED: Using SQLAlchemy engine
        feedback_df = pd.read_sql_query('''
        SELECT uf.*, ql.query_category 
        FROM user_feedback uf
        JOIN query_logs ql ON uf.query_id = ql.id
        WHERE uf.timestamp >= ? AND uf.timestamp <= ?
        ''', engine, params=(start_date, end_date))
        
        # Feature usage - FIXED: Using SQLAlchemy engine
        feature_df = pd.read_sql_query('''
        SELECT * FROM feature_usage 
        WHERE timestamp >= ? AND timestamp <= ?
        ''', engine, params=(start_date, end_date))
        
        # Close engine connection
        engine.dispose()
        
        # Fix datetime conversion
        daily_queries = {}
        if not query_df.empty:
            # Convert timestamp to datetime
            query_df['timestamp'] = pd.to_datetime(query_df['timestamp'])
            daily_queries = query_df.groupby(query_df['timestamp'].dt.date).size().to_dict()
        
        # Calculate key metrics (rest of your existing code remains the same)
        analytics_data = {
            'total_queries': len(query_df),
            'unique_users': query_df['user_id'].nunique() if not query_df.empty else 0,
            'avg_response_time': query_df['response_time_seconds'].mean() if not query_df.empty else 0,
            'avg_rating': feedback_df['rating'].mean() if not feedback_df.empty else 0,
            'error_rate': (query_df['error_occurred'].sum() / len(query_df) * 100) if not query_df.empty else 0,
            'queries_with_attachments': query_df['has_attachments'].sum() if not query_df.empty else 0,
            'query_categories': query_df['query_category'].value_counts().to_dict() if not query_df.empty else {},
            'daily_queries': daily_queries,
            'performance_modes': query_df['performance_mode'].value_counts().to_dict() if not query_df.empty else {},
            'feedback_scores': {
                'usefulness': feedback_df['usefulness_score'].mean() if not feedback_df.empty else 0,
                'accuracy': feedback_df['accuracy_score'].mean() if not feedback_df.empty else 0,
                'clarity': feedback_df['clarity_score'].mean() if not feedback_df.empty else 0
            },
            'raw_data': {
                'queries': query_df,
                'feedback': feedback_df,
                'features': feature_df
            }
        }
        
        return analytics_data


    def get_feedback_with_comments(self, days: int = 30) -> pd.DataFrame:
        """Get feedback comments using SQLAlchemy to avoid pandas warnings."""
        engine = create_engine(f'sqlite:///{self.db_path}')
        
        feedback_with_comments = pd.read_sql_query('''
        SELECT 
            uf.timestamp,
            uf.rating,
            uf.usefulness_score,
            uf.accuracy_score, 
            uf.clarity_score,
            uf.feedback_text,
            ql.query_category,
            ql.query_text,
            ql.response_time_seconds
        FROM user_feedback uf
        JOIN query_logs ql ON uf.query_id = ql.id
        WHERE uf.feedback_text IS NOT NULL AND uf.feedback_text != ''
        AND uf.timestamp >= ?
        ORDER BY uf.timestamp DESC
        LIMIT 50
        ''', engine, params=(datetime.now() - timedelta(days=days),))
        
        engine.dispose()
        return feedback_with_comments

    def get_feedback_trends(self, days: int = 30) -> pd.DataFrame:
        """Get feedback trends using SQLAlchemy to avoid pandas warnings."""
        engine = create_engine(f'sqlite:///{self.db_path}')
        
        all_feedback = pd.read_sql_query('''
        SELECT 
            uf.timestamp,
            uf.rating,
            uf.usefulness_score,
            uf.accuracy_score,
            uf.clarity_score,
            ql.query_category
        FROM user_feedback uf
        JOIN query_logs ql ON uf.query_id = ql.id
        WHERE uf.timestamp >= ?
        ORDER BY uf.timestamp
        ''', engine, params=(datetime.now() - timedelta(days=days),))
        
        engine.dispose()
        return all_feedback



# Streamlit component for user feedback collection
def show_feedback_widget(tracker: PerformanceTracker):
    """Display user feedback collection widget"""
    
    if 'current_query_id' not in st.session_state:
        return
    
    query_id = st.session_state.current_query_id
    
    with st.expander("📊 How was this response?", expanded=False):
        st.markdown("**Help us improve by rating this response:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            usefulness = st.select_slider(
                "Usefulness",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: ["😔", "😐", "🙂", "😊", "😍"][x-1]
            )
        
        with col2:
            accuracy = st.select_slider(
                "Accuracy",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: ["❌", "⚠️", "✅", "💯", "🎯"][x-1]
            )
        
        with col3:
            clarity = st.select_slider(
                "Clarity",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: ["🤔", "😕", "👍", "✨", "🔥"][x-1]
            )
        
        feedback_text = st.text_area(
            "Additional feedback (optional):",
            placeholder="Any specific comments or suggestions?",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Submit Feedback", type="primary", use_container_width=True):
                overall_rating = round((usefulness + accuracy + clarity) / 3)
                tracker.log_user_feedback(
                    query_id=query_id,
                    rating=overall_rating,
                    feedback_type="detailed",
                    feedback_text=feedback_text if feedback_text else None,
                    usefulness_score=usefulness,
                    accuracy_score=accuracy,
                    clarity_score=clarity
                )
                st.success("Thank you for your feedback!")
                # Clear the query_id to prevent duplicate submissions
                del st.session_state.current_query_id
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip", use_container_width=True):
                # Clear without logging
                del st.session_state.current_query_id
                st.rerun()

# Analytics Dashboard
def show_analytics_dashboard(tracker: PerformanceTracker):
    """Display comprehensive analytics dashboard with feedback comments"""
    
    st.markdown("## 📊 Lumen Navigator Analytics Dashboard")
    
    # Time range selector - DEFINE 'days' FIRST
    col1, col2 = st.columns([3, 1])
    with col1:
        days = st.selectbox(
            "Time Range:",
            options=[7, 30, 90, 365],
            index=1,
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Get analytics data
    data = tracker.get_analytics_dashboard_data(days)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            value=data['total_queries'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Unique Users",
            value=data['unique_users'],
            delta=None
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            value=f"{data['avg_response_time']:.2f}s",
            delta=None
        )
    
    with col4:
        st.metric(
            "User Satisfaction",
            value=f"{data['avg_rating']:.1f}/5" if data['avg_rating'] > 0 else "No ratings yet",
            delta=None
        )
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        if data['query_categories']:
            fig = px.pie(
                values=list(data['query_categories'].values()),
                names=list(data['query_categories'].keys()),
                title="Query Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if data['daily_queries']:
            dates = list(data['daily_queries'].keys())
            counts = list(data['daily_queries'].values())
            fig = px.line(
                x=dates, y=counts,
                title="Daily Query Volume",
                labels={'x': 'Date', 'y': 'Number of Queries'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # System Health Indicators
    st.markdown("### 🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        error_rate = data['error_rate']
        color = "red" if error_rate > 5 else "orange" if error_rate > 1 else "green"
        st.metric(
            "Error Rate",
            value=f"{error_rate:.1f}%",
            delta=None
        )
    
    with col2:
        attachment_usage = (data['queries_with_attachments'] / data['total_queries'] * 100) if data['total_queries'] > 0 else 0
        st.metric(
            "Queries with Attachments",
            value=f"{attachment_usage:.1f}%",
            delta=None
        )
    
    with col3:
        if data['feedback_scores']['usefulness'] > 0:
            st.metric(
                "Avg Usefulness Score",
                value=f"{data['feedback_scores']['usefulness']:.1f}/5",
                delta=None
            )
    
    # Recent Queries Table
    if not data['raw_data']['queries'].empty:
        st.markdown("### 🔍 Recent Queries")
        
        recent_queries = data['raw_data']['queries'].head(10)[
            ['timestamp', 'query_category', 'query_length', 'response_time_seconds', 'error_occurred']
        ].copy()
        
        recent_queries['timestamp'] = pd.to_datetime(recent_queries['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent_queries.columns = ['Time', 'Category', 'Query Length', 'Response Time (s)', 'Had Error']
        
        st.dataframe(recent_queries, use_container_width=True)
    
    # NOW ADD THE FEEDBACK COMMENTS SECTION
    st.markdown("### 💬 User Feedback Comments")
    
    # Get feedback with comments
    feedback_with_comments = tracker.get_feedback_with_comments(days)
    
    if not feedback_with_comments.empty:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rating_filter = st.selectbox(
                "Filter by Rating:",
                options=["All"] + list(range(1, 6)),
                format_func=lambda x: f"{x} stars" if x != "All" else "All ratings"
            )
        
        with col2:
            category_filter = st.selectbox(
                "Filter by Category:",
                options=["All"] + list(feedback_with_comments['query_category'].unique())
            )
        
        with col3:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment:",
                options=["All", "Positive (4-5)", "Neutral (3)", "Negative (1-2)"]
            )
        
        # Apply filters
        filtered_feedback = feedback_with_comments.copy()
        
        if rating_filter != "All":
            filtered_feedback = filtered_feedback[filtered_feedback['rating'] == rating_filter]
        
        if category_filter != "All":
            filtered_feedback = filtered_feedback[filtered_feedback['query_category'] == category_filter]
        
        if sentiment_filter != "All":
            if sentiment_filter == "Positive (4-5)":
                filtered_feedback = filtered_feedback[filtered_feedback['rating'] >= 4]
            elif sentiment_filter == "Neutral (3)":
                filtered_feedback = filtered_feedback[filtered_feedback['rating'] == 3]
            elif sentiment_filter == "Negative (1-2)":
                filtered_feedback = filtered_feedback[filtered_feedback['rating'] <= 2]
        
        st.write(f"**{len(filtered_feedback)} feedback comments found**")
        
        # Display feedback comments using Streamlit native components
        for idx, row in filtered_feedback.head(20).iterrows():
            timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
            
            # Determine sentiment
            if row['rating'] >= 4:
                sentiment_emoji = "😊"
                sentiment_color = "green"
            elif row['rating'] == 3:
                sentiment_emoji = "😐"
                sentiment_color = "orange"
            else:
                sentiment_emoji = "😞"
                sentiment_color = "red"
            
            # Create container for each feedback
            with st.container():
                # Header row with badges
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{sentiment_emoji} {row['rating']}/5 stars**")
                
                with col2:
                    st.markdown(f"📂 **{row['query_category'].title()}**")
                
                with col3:
                    st.markdown(f"🕒 {timestamp}")
                
                with col4:
                    # Response time
                    st.markdown(f"⚡ {row['response_time_seconds']:.1f}s")
                
                # Feedback text in a quote box
                st.markdown(f"> **User Feedback:** \"{row['feedback_text']}\"")
                
                # Scores in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="📊 Usefulness",
                        value=f"{row['usefulness_score']}/5"
                    )
                
                with col2:
                    st.metric(
                        label="🎯 Accuracy", 
                        value=f"{row['accuracy_score']}/5"
                    )
                
                with col3:
                    st.metric(
                        label="💡 Clarity",
                        value=f"{row['clarity_score']}/5"
                    )
                
                # Original query in expander
                with st.expander("📝 View Original Query"):
                    st.write(f"**Query:** {row['query_text']}")
                
                # Add separator
                st.markdown("---")
        
        # Summary statistics for filtered feedback
        if len(filtered_feedback) > 0:
            st.markdown("#### 📈 Feedback Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rating = filtered_feedback['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
            
            with col2:
                positive_percent = (filtered_feedback['rating'] >= 4).sum() / len(filtered_feedback) * 100
                st.metric("Positive Feedback", f"{positive_percent:.1f}%")
            
            with col3:
                avg_usefulness = filtered_feedback['usefulness_score'].mean()
                st.metric("Avg Usefulness", f"{avg_usefulness:.1f}/5")
            
            with col4:
                avg_clarity = filtered_feedback['clarity_score'].mean()
                st.metric("Avg Clarity", f"{avg_clarity:.1f}/5")
    
    else:
        st.info("📝 No written feedback comments yet. Encourage users to share their thoughts!")
    
    # ADD FEEDBACK TRENDS ANALYSIS
    st.markdown("### 📊 Feedback Trends")
    
    # Get all feedback for trend analysis using the fixed method
    all_feedback = tracker.get_feedback_trends(days)
    
    if not all_feedback.empty:
        # Convert timestamp and create daily averages
        all_feedback['timestamp'] = pd.to_datetime(all_feedback['timestamp'])
        all_feedback['date'] = all_feedback['timestamp'].dt.date
        
        daily_ratings = all_feedback.groupby('date').agg({
            'rating': 'mean',
            'usefulness_score': 'mean',
            'accuracy_score': 'mean', 
            'clarity_score': 'mean'
        }).reset_index()
        
        # Create trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_ratings['date'], 
            y=daily_ratings['rating'],
            mode='lines+markers',
            name='Overall Rating',
            line=dict(color='#3b82f6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_ratings['date'], 
            y=daily_ratings['usefulness_score'],
            mode='lines',
            name='Usefulness',
            line=dict(color='#10b981')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_ratings['date'], 
            y=daily_ratings['accuracy_score'],
            mode='lines', 
            name='Accuracy',
            line=dict(color='#f59e0b')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_ratings['date'], 
            y=daily_ratings['clarity_score'],
            mode='lines',
            name='Clarity', 
            line=dict(color='#8b5cf6')
        ))
        
        fig.update_layout(
            title="Daily Average Ratings Trends",
            xaxis_title="Date",
            yaxis_title="Rating (1-5)",
            yaxis=dict(range=[1, 5]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.markdown("### 📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Feedback Data (CSV)"):
            if not feedback_with_comments.empty:
                csv_data = feedback_with_comments.to_csv(index=False)
                st.download_button(
                    label="📥 Download Feedback CSV",
                    data=csv_data,
                    file_name=f"lumen_feedback_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No feedback data to export")
    
    with col2:
        if st.button("📈 Export Analytics Summary"):
            summary_data = {
                "report_date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "time_period_days": days,
                **data
            }
            
            summary_json = json.dumps(summary_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Summary JSON", 
                data=summary_json,
                file_name=f"lumen_analytics_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

# Integration helper functions
def track_query_performance(func):
    """Decorator to automatically track query performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log successful query (you'll need to adapt this to your specific function)
            if hasattr(st.session_state, 'performance_tracker'):
                # This would need to be customized based on your specific function signature
                pass
            
            return result
        except Exception as e:
            end_time = time.time()
            
            # Log error
            if hasattr(st.session_state, 'performance_tracker'):
                st.session_state.performance_tracker.log_system_metric(
                    "error", "query_processing_error", end_time - start_time,
                    {"error": str(e)}
                )
            
            raise
    
    return wrapper
