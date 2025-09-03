# app.py - Enhanced with Website-Matching Design

import os
import sys
import warnings
import streamlit as st
import time
from datetime import datetime
from PIL import Image
import io
import re
from typing import Dict, Any, Optional, List
from performance_tracker import PerformanceTracker, show_feedback_widget, show_analytics_dashboard
from beta_access_system import BetaAccessManager
from beta_ui_components import show_beta_access_gate, show_beta_admin_panel, track_beta_query_usage

# STREAMLINED AUTH0 IMPORT
from streamlined_auth0 import StreamlitAuth0

# Your existing imports (keep these the same)
try:
    from rag_system import HybridRAGSystem as EnhancedRAGSystem, create_hybrid_rag_system as create_rag_system
    RAG_SYSTEM_AVAILABLE = True
except ImportError:
    RAG_SYSTEM_AVAILABLE = False
    print("⚠️ RAG system files not found - you'll need to copy them from your original project")

try:
    from phase1_enhancements import enhance_your_rag_system, test_enhancement_integration
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False

try:
    from phase2_metadata_chunking import integrate_phase2_with_existing_rag, test_phase2_integration
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Lumen Navigator - Children's Home Management",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up environment and suppress warnings
def setup_environment():
    """Setup environment variables and suppress warnings"""
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '3'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Initialize performance tracker
    if 'performance_tracker' not in st.session_state:
        st.session_state.performance_tracker = PerformanceTracker()

def setup_sqlite():
    """SQLite setup for compatibility"""
    try:
        import pysqlite3
        import sys
        sys.modules['sqlite3'] = pysqlite3
    except ImportError:
        pass  # Use standard sqlite3
    except Exception as e:
        # Log the error but don't crash
        print(f"SQLite setup warning: {e}")
        pass

# RAG system initialization (only if files are available)
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system if available"""
    if not RAG_SYSTEM_AVAILABLE:
        return None
        
    try:
        # Check what directories exist (no debug output)
        faiss_exists = os.path.exists("faiss_index")
        indexes_exists = os.path.exists("indexes")
        
        # Check for actual vector database files
        has_vector_db = False
        if indexes_exists:
            google_index_exists = os.path.exists("indexes/google_index")
            openai_index_exists = os.path.exists("indexes/openai_index")
            has_vector_db = google_index_exists or openai_index_exists
        elif faiss_exists:
            faiss_files = os.listdir("faiss_index") if faiss_exists else []
            has_vector_db = any(f.endswith(('.faiss', '.pkl')) for f in faiss_files)
        
        if not has_vector_db:
            st.warning("Vector database not found - some features may be limited")
            return None
        
        # Create the system without debug output
        rag_system = create_rag_system()
        return rag_system
        
    except Exception as e:
        st.error(f"RAG System Error: {str(e)}")
        return None

# Enhanced CSS Styles matching website design
@st.cache_data
def get_enhanced_css():
    return """
    <style>
    /* Import Google Fonts - Poppins (same as website) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@600;700&display=swap');
    
    /* Global styling to match website */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */
    
    /* Main header with gradient matching website */
    .hero-gradient {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.02em;
        color: white;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    
    .hero-welcome {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Card styling matching website */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .service-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Question and result styling */
    .question-container {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    .question-title {
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .result-header {
        color: #1f2937;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button styling matching website */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Secondary button style */
    .secondary-button {
        background: white !important;
        color: #3b82f6 !important;
        border: 2px solid #3b82f6 !important;
    }
    
    .secondary-button:hover {
        background: #f8fafc !important;
        transform: translateY(-1px);
    }
    
    /* Orange accent buttons (matching website orange) */
    .orange-button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .orange-button:hover {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
    }
    
    /* Success/status styling */
    .success-badge {
        background: #dcfce7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .warning-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .info-badge {
        background: #dbeafe;
        color: #1e40af;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    /* Text area and input styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Poppins', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    /* Footer styling */
    .app-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        padding: 2rem 1rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
        background: #f9fafb;
        border-radius: 12px;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .result-container {
            padding: 1.5rem;
        }
    }

    /* Analytics dashboard styling */
        .analytics-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 1rem 0;
        }
        
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        
        .health-indicator-good {
            color: #059669;
            font-weight: 600;
        }
        
        .health-indicator-warning {
            color: #d97706;
            font-weight: 600;
        }
        
        .health-indicator-error {
            color: #dc2626;
            font-weight: 600;
        }
    </style>
    """

# Initialize UI state
def initialize_ui_state():
    """Initialize UI state"""
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = {
            'current_result': None,
            'current_question': "",
            'show_sources': False,
            'question_counter': 0
        }

# Enhanced header function
def show_enhanced_header(user_info=None):
    """Show enhanced header matching website design"""
    name = user_info.get('name', 'Professional') if user_info else 'Professional'
    
    st.markdown(f"""
    <div class="hero-gradient">
        <div class="hero-title">🏠 Lumen Navigator</div>
        <div class="hero-subtitle">Smart technology that takes the complexity out of children's home management</div>
        <div class="hero-welcome">Welcome, {name}! Ready to get expert guidance?</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced welcome interface
def show_enhanced_welcome_interface():
    """Show enhanced welcome interface when RAG system is not available"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 System Setup Required</h3>
            <p style="color: #6b7280; margin-bottom: 1rem;">Authentication is working perfectly! To unlock the full power of Lumen Navigator, complete these steps:</p>
            
            <div style="background: #f3f4f6; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <ol style="margin: 0; padding-left: 1.5rem; color: #374151;">
                    <li style="margin: 0.5rem 0;"><strong>Copy RAG System Files:</strong> rag_system.py, smart_query_router.py, etc.</li>
                    <li style="margin: 0.5rem 0;"><strong>Setup Vector Database:</strong> FAISS index with your knowledge base</li>
                    <li style="margin: 0.5rem 0;"><strong>Restart Application:</strong> Full AI capabilities will activate</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="service-card">
            <h4>🎯 Full System Capabilities</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <strong style="color: #f59e0b;">🧠 AI Analysis:</strong><br>
                    <span style="color: #6b7280; font-size: 0.9rem;">Intelligent guidance and recommendations</span>
                </div>
                <div>
                    <strong style="color: #f59e0b;">📊 Ofsted Reports:</strong><br>
                    <span style="color: #6b7280; font-size: 0.9rem;">Automated analysis and pathways</span>
                </div>
                <div>
                    <strong style="color: #f59e0b;">🛡️ Safeguarding:</strong><br>
                    <span style="color: #6b7280; font-size: 0.9rem;">Framework guidance and risk assessment</span>
                </div>
                <div>
                    <strong style="color: #f59e0b;">📋 Compliance:</strong><br>
                    <span style="color: #6b7280; font-size: 0.9rem;">Policy checking and documentation</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);">
            <h4 style="color: #0369a1;">🔐 Security Status</h4>
            <div class="success-badge" style="width: 100%; text-align: center; margin: 1rem 0;">
                ✅ Authentication Active
            </div>
            <ul style="list-style: none; padding: 0; margin: 1rem 0;">
                <li style="padding: 0.25rem 0; color: #0369a1;">🔒 Auth0 Professional</li>
                <li style="padding: 0.25rem 0; color: #0369a1;">🛡️ Secure Sessions</li>
                <li style="padding: 0.25rem 0; color: #0369a1;">📝 Activity Logging</li>
                <li style="padding: 0.25rem 0; color: #0369a1;">🎫 Access Controls</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Test interface
    st.markdown("---")
    st.markdown("### 🧪 Test Interface")
    
    test_question = st.text_area(
        "Test the system (basic response mode):",
        placeholder="Try asking: 'What are the key areas for Ofsted inspection preparation?' or 'How should we handle a safeguarding concern?'",
        height=100,
        help="Enter any question about children's home management to test the interface"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💬 Get Basic Guidance", type="primary", use_container_width=True):
            if test_question:
                with st.spinner("Processing your test question..."):
                    time.sleep(1)  # Simulate processing
                
                st.success("✅ Authentication and interface working perfectly!")
                
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-header">🤖 Test Response</div>
                    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                        <strong>Your question:</strong> "{test_question}"<br><br>
                        <strong>System status:</strong> Interface and authentication are fully operational! 
                        Once you complete the RAG system setup, this will provide comprehensive, 
                        AI-powered guidance drawing from specialized children's home management knowledge bases, 
                        regulatory frameworks, and best practice guidelines.
                        <br><br>
                        <strong>Next step:</strong> Complete the setup to unlock intelligent document analysis, 
                        policy compliance checking, and personalized recommendations.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a test question to proceed.")

# Enhanced full interface
def show_enhanced_full_interface():
    """Show enhanced full interface when RAG system is available"""
    
    st.markdown("""
    <div class="feature-card">
        <h3>💬 Ask Your Question</h3>
        <p style="color: #6b7280;">Describe your situation or ask any question about children's home management. 
        Our AI system will provide expert guidance based on comprehensive knowledge bases and regulatory frameworks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question input with integrated file upload
    col_q1, col_q2 = st.columns([4, 1])

    with col_q1:
        user_question = st.text_area(
            "Enter your question or describe your situation:",
            value="",
            placeholder="Examples:\n• 'What should we focus on for our next Ofsted inspection?'\n• 'How do we handle a safeguarding concern involving a young person?'\n• 'Review our medication policy for compliance issues'\n• 'What training do staff need for trauma-informed care?'",
            height=120,
            key=f"question_input_{st.session_state.ui_state.get('question_counter', 0)}",
            help="Ask about operations, compliance, policies, training, or any aspect of children's home management"
        )

    with col_q2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # SINGLE file uploader
        uploaded_files_all = st.file_uploader(
            "📎 Attach Files",
            type=['pdf', 'docx', 'txt', 'md', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload documents, policies, Ofsted reports, or facility images",
            label_visibility="collapsed",
            key="main_file_uploader"
        )
        
        # SINGLE file separation function (KEEP ONLY THIS ONE)
        uploaded_files = []
        uploaded_images = []
        
        if uploaded_files_all:
            for file in uploaded_files_all:
                file_ext = file.name.lower()
                if file_ext.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    uploaded_images.append(file)
                elif file_ext.endswith(('.pdf', '.docx', '.txt', '.md', '.doc')):
                    uploaded_files.append(file)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file.name}")
    
    # SINGLE file count display (KEEP ONLY THIS ONE)
    if uploaded_files_all:
        file_counts = []
        if uploaded_files:
            file_counts.append(f"📄 {len(uploaded_files)} document(s)")
        if uploaded_images:
            file_counts.append(f"🖼️ {len(uploaded_images)} image(s)")
        
        display_text = " • ".join(file_counts) if file_counts else f"📁 {len(uploaded_files_all)} file(s)"
        
        st.markdown(f"""
        <div style="background: #dcfce7; color: #166534; padding: 0.5rem; border-radius: 6px; font-size: 0.8rem; text-align: center; margin-top: 0.5rem;">
            {display_text}
        </div>
        """, unsafe_allow_html=True)
    
    # SINGLE action button (KEEP ONLY THIS ONE)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🧠 Get Expert AI Guidance", type="primary", use_container_width=True, key="main_guidance_button_single"):
            if user_question.strip() or uploaded_files_all:
                process_enhanced_request(user_question, uploaded_files, uploaded_images)
            else:
                st.warning("⚠️ Please enter a question or upload files for analysis")

def process_enhanced_request(question, uploaded_files=None, uploaded_images=None):
    """Process user request with enhanced UI feedback and performance tracking"""
    
    # Get tracker instance
    tracker = st.session_state.performance_tracker
    
    # Updated tracking with image support
    total_attachments = (len(uploaded_files) if uploaded_files else 0) + (len(uploaded_images) if uploaded_images else 0)
    
    # Track feature usage
    tracker.log_feature_usage(
        feature_name="query_submission",
        action="submit_query",
        context={
            "has_attachments": total_attachments > 0,
            "attachment_count": total_attachments,
            "document_count": len(uploaded_files) if uploaded_files else 0,
            "image_count": len(uploaded_images) if uploaded_images else 0,
            "query_length": len(question),
            "performance_mode": st.session_state.get('performance_mode', 'balanced')
        }
    )
    
    try:
        # Enhanced loading state
        progress_placeholder = st.empty()
        
        with progress_placeholder:
            st.markdown("""
            <div class="feature-card loading">
                <div style="text-align: center;">
                    <h4>🧠 AI Analysis in Progress</h4>
                    <p>Processing your question and analyzing uploaded content...</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Generating expert guidance..."):
            # Start performance timing
            start_time = time.time()
            
            time.sleep(1)  # Brief pause for UX
            
            # Use RAG system if available
            if st.session_state.get('rag_system'):
                result = st.session_state.rag_system.query(
                    question=question,
                    k=5,
                    response_style="standard",
                    performance_mode=st.session_state.get('performance_mode', 'balanced'),
                    uploaded_files=uploaded_files,
                    uploaded_images=uploaded_images
                )
                
                response_time = time.time() - start_time
                
                query_id = tracker.log_query(
                    query_text=question,
                    response_text=result.get("answer", ""),
                    response_time=response_time,
                    sources=result.get("sources", []),
                    attachments={"documents": uploaded_files, "images": uploaded_images},
                    performance_mode=st.session_state.get('performance_mode', 'balanced'),
                    error_info=None
                )
                
            else:
                # Fallback response
                response_time = time.time() - start_time
                result = {
                    "answer": f"Based on your question: '{question}' - This is a fallback response.",
                    "sources": [],
                    "metadata": {"note": "Fallback response"}
                }
        
        progress_placeholder.empty()
        
        if result and result.get("answer"):
            st.session_state.ui_state.update({
                'current_result': result,
                'current_question': question,
                'show_sources': False
            })
            st.success("✅ Expert guidance generated successfully!")
            st.rerun()
        else:
            st.error("❌ Sorry, I couldn't generate a response. Please try rephrasing your question.")
            
    except Exception as e:
        st.error(f"❌ An error occurred while processing your request: {str(e)}")

def show_enhanced_result_display():
    """Display results with enhanced styling and feedback collection"""
    result = st.session_state.ui_state['current_result']
    question = st.session_state.ui_state['current_question']
    tracker = st.session_state.performance_tracker
    
    # Enhanced question display
    st.markdown(f"""
    <div class="question-container">
        <div class="question-title">🤔 Your Question</div>
        <div style="font-size: 1rem; line-height: 1.5;">{question}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced answer display
    st.markdown(f"""
    <div class="result-container">
        <div class="result-header">
            🧠 Expert AI Guidance
        </div>
        <div style="line-height: 1.6; color: #374151;">
            {result["answer"]}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ADD FEEDBACK WIDGET HERE
    show_feedback_widget(tracker)
    
    # Enhanced action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Ask New Question", type="primary"):
            # Track feature usage
            tracker.log_feature_usage(
                feature_name="navigation",
                action="new_question",
                context={"from_results_page": True}
            )
            
            st.session_state.ui_state.update({
                'current_result': None,
                'current_question': "",
                'question_counter': st.session_state.ui_state.get('question_counter', 0) + 1
            })
            st.rerun()
    
    with col2:
        if st.button("📚 View Sources"):
            # Track feature usage
            tracker.log_feature_usage(
                feature_name="sources",
                action="view_sources",
                context={"sources_count": len(result.get("sources", []))}
            )
            
            sources = result.get("sources", [])
            if sources:
                st.markdown(f"""
                <div class="info-badge">
                    📖 {len(sources)} source documents referenced
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Source Details"):
                    for i, source in enumerate(sources[:5], 1):
                        st.write(f"**Source {i}:** {source.get('title', 'Document')}")
                        if 'content' in source:
                            st.write(f"_{source['content'][:200]}..._")
                        st.write("---")
            else:
                st.info("📄 No specific source documents were referenced for this response")
    
    with col3:
        if st.button("💾 Download Report"):
            # Track feature usage
            tracker.log_feature_usage(
                feature_name="export",
                action="download_report",
                context={"response_length": len(result["answer"])}
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = f"""# Lumen Navigator Expert Guidance Report

## Question
{question}

## AI Response
{result["answer"]}

## Metadata
- **Generated:** {datetime.now().strftime('%d %B %Y at %H:%M')}
- **System:** Lumen Navigator AI
- **Sources Referenced:** {len(result.get("sources", []))}

---
*This report was generated by Lumen Navigator - Professional Children's Home Management System*
"""
            st.download_button(
                label="📄 Download",
                data=report_content,
                file_name=f"Lumen_Guidance_{timestamp}.md",
                mime="text/markdown"
            )
    
    with col4:
        if st.button("📧 Share Results"):
            # Track feature usage
            tracker.log_feature_usage(
                feature_name="sharing",
                action="share_results",
                context={"sharing_method": "email_info"}
            )
            
            st.info("📧 Sharing functionality - contact your system administrator for email integration setup")

def show_enhanced_sidebar():
    """Enhanced sidebar with system status, settings, and admin-only analytics"""
    with st.sidebar:
        st.markdown("### ⚡ System Performance")
        
        if st.session_state.get('rag_system'):
            performance_mode = st.selectbox(
                "Response Mode:",
                ["fast", "balanced", "comprehensive"],
                index=1,
                help="Choose speed vs comprehensiveness trade-off"
            )
            
            # Track performance mode changes
            if st.session_state.get('performance_mode') != performance_mode:
                st.session_state.performance_tracker.log_feature_usage(
                    feature_name="settings",
                    action="change_performance_mode",
                    context={"new_mode": performance_mode, "old_mode": st.session_state.get('performance_mode')}
                )
            
            st.session_state['performance_mode'] = performance_mode
            
            st.markdown("""
            <div class="status-card">
                <strong>Current Mode:</strong><br>
                <span style="color: #059669;">● """ + performance_mode.title() + """</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔍 System Status")
        
        # System status with enhanced styling
        if st.session_state.get('rag_system'):
            st.markdown('<div class="success-badge">✅ Full RAG System Ready</div>', unsafe_allow_html=True)
        elif RAG_SYSTEM_AVAILABLE:
            st.markdown('<div class="warning-badge">⚠️ RAG System Available (Database Missing)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-badge">ℹ️ Basic Mode (RAG Files Missing)</div>', unsafe_allow_html=True)
        
        # Component status
        st.markdown(f"""
        <div class="status-card">
            <div style="margin: 0.5rem 0;">
                <strong>Components:</strong><br>
                🧠 Phase 1: {'✅ Active' if PHASE1_AVAILABLE else '❌ Missing'}<br>
                📊 Phase 2: {'✅ Active' if PHASE2_AVAILABLE else '❌ Missing'}<br>
                🔐 Auth0: ✅ Active<br>
                🎨 UI: ✅ Enhanced<br>
                📈 Analytics: ✅ Active
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Admin-only analytics access
        st.markdown("---")
        
        # Check if user is admin (multiple verification methods)
        is_admin = False
        
        # Method 1: Environment variable (for development)
        if os.getenv('LUMEN_ADMIN_MODE') == 'true':
            is_admin = True
        
        # Method 2: Check user email domain (if using Auth0)
        try:
            user_info = st.session_state.get('user_info', {})
            user_email = user_info.get('email', '')
            # Add your authorized admin emails here
            admin_emails = [
                'garybrooks0@gmail.com', 
                'gbrooks@lumenwayhomes.org.uk',
                'analytics@lumenwayhomes.org.uk'
            ]
            if user_email.endswith('@lumenwayhomes.org.uk') or user_email in admin_emails:
                is_admin = True
        except:
            pass
        
        # Method 3: Secret admin key authentication
        if st.session_state.get('admin_authenticated', False):
            is_admin = True
        
        # Show appropriate interface based on admin status
        if is_admin:
            st.markdown("### Analytics & Administration")
            st.markdown("""
            <div style="background: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="color: #0369a1; font-size: 0.85rem;">
                    🔧 <strong>Administrator Access</strong><br>
                    Performance tracking and beta management
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📈 Analytics", use_container_width=True):
                    st.session_state.show_analytics = True
                    st.rerun()
            
            with col2:
                if st.button("🧪 Beta Admin", use_container_width=True):
                    st.session_state.show_beta_admin = True
                    st.rerun()

            if st.button("🔍 Test Beta System", use_container_width=True):
                st.write("### 🧪 Beta System Integration Test")
                
                try:
                    # Test database creation
                    beta_manager = BetaAccessManager()
                    st.success("✅ Database initialized successfully")
                    
                    # Test admin dashboard data
                    dashboard_data = beta_manager.get_admin_dashboard_data()
                    st.success("✅ Admin dashboard data accessible")
                    st.write("**Dashboard Data:**")
                    st.json(dashboard_data)
                    
                    # Test user access check (with current user)
                    user_info = st.session_state.get('user_info', {})
                    access_info = beta_manager.check_user_access(user_info)
                    st.success("✅ User access check working")
                    st.write("**Current User Access:**")
                    st.json(access_info)
                    
                except Exception as e:
                    st.error(f"❌ Integration test failed: {e}")    
            
            # Quick stats for admin
            try:
                tracker = st.session_state.performance_tracker
                # Get basic stats for last 7 days
                data = tracker.get_analytics_dashboard_data(7)
                
                st.markdown("**Quick Stats (7 days):**")
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #6b7280;">
                    • Queries: {data['total_queries']}<br>
                    • Users: {data['unique_users']}<br>
                    • Avg Response: {data['avg_response_time']:.1f}s<br>
                    • Satisfaction: {data['avg_rating']:.1f}/5 
                </div>
                """, unsafe_allow_html=True)
            except:
                st.caption("Analytics data loading...")
                
        else:
            # Non-admin interface - show limited admin access
            st.markdown("### 🔧 System Access")
            
            # Show admin login option for development/emergency access
            if st.button("🔑 Admin Login", use_container_width=True):
                if 'show_admin_login' not in st.session_state:
                    st.session_state.show_admin_login = True
                else:
                    st.session_state.show_admin_login = not st.session_state.show_admin_login
                st.rerun()
            
            # Admin login form (hidden by default)
            if st.session_state.get('show_admin_login', False):
                st.markdown("**Admin Authentication:**")
                admin_key = st.text_input(
                    "Admin Key:", 
                    type="password", 
                    key="admin_key_input",
                    placeholder="Enter admin access key"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Login", key="admin_login_btn"):
                        # Change this to your secure admin key
                        if admin_key == "lumen_admin_2024_secure":
                            st.session_state.admin_authenticated = True
                            st.session_state.show_admin_login = False
                            st.success("Admin access granted!")
                            time.sleep(1)
                            st.rerun()
                        elif admin_key:
                            st.error("Invalid admin key")
                
                with col2:
                    if st.button("Cancel", key="admin_cancel_btn"):
                        st.session_state.show_admin_login = False
                        st.rerun()
        
        # Setup guide for non-RAG systems
        if not st.session_state.get('rag_system'):
            st.markdown("---")
            st.markdown("### 🚀 Setup Guide")
            st.markdown("""
            <div style="font-size: 0.9rem; color: #6b7280;">
                <strong>Quick Setup:</strong><br>
                1. Copy RAG system files<br>
                2. Setup FAISS database<br>
                3. Restart application<br>
                <br>
                <strong>Need help?</strong><br>
                Contact your system administrator
            </div>
            """, unsafe_allow_html=True)


# ===== ADD THESE UTILITY FUNCTIONS =====
def track_user_session_start():
    """Track when a user session starts"""
    if 'session_tracked' not in st.session_state:
        tracker = st.session_state.performance_tracker
        tracker.log_feature_usage(
            feature_name="session",
            action="session_start",
            context={
                "user_agent": st.get_option("browser.gatherUsageStats"),  # Limited browser info
                "timestamp": datetime.now().isoformat()
            }
        )
        st.session_state.session_tracked = True

def track_page_view(page_name: str):
    """Track page views within the app"""
    tracker = st.session_state.performance_tracker
    tracker.log_feature_usage(
        feature_name="navigation",
        action="page_view",
        context={"page": page_name}
    )

def test_beta_system_integration():
    """Test function to verify beta system is working"""
    st.write("Testing beta system integration...")
    
    try:
        # Test database creation
        beta_manager = BetaAccessManager()
        st.success("✅ Database initialized successfully")
        
        # Test admin dashboard data
        dashboard_data = beta_manager.get_admin_dashboard_data()
        st.success("✅ Admin dashboard data accessible")
        
        # Test user access check (with dummy user)
        test_user = {"sub": "test123", "email": "test@example.com", "name": "Test User"}
        access_info = beta_manager.check_user_access(test_user)
        st.success("✅ User access check working")
        st.json(access_info)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Integration test failed: {e}")
        return False

# ===== ADD TO MAIN APP FLOW =====


def main():
    """Enhanced main application with beta access control"""
    # Initialize environment (keep existing code)
    setup_environment()
    setup_sqlite()
    
    # Load enhanced CSS (keep existing)
    st.markdown(get_enhanced_css(), unsafe_allow_html=True)
    
    # STREAMLINED AUTH0 AUTHENTICATION (keep existing)
    auth = StreamlitAuth0()
    if not auth.require_authentication():
        st.stop()
    
    # GET USER INFO (keep existing)
    user_info = auth.get_user_info()
    
    # Check that we actually have user info before proceeding
    if user_info is None:
        st.error("Unable to retrieve user information. Please try logging in again.")
        st.stop()

    # === NEW: BETA ACCESS GATE ===
    access_granted, access_info = show_beta_access_gate(user_info)
    if not access_granted:
        st.stop()  # Show beta blocking page and stop here
    
    # === EXISTING CODE CONTINUES (only if access granted) ===
    
    # Initialize UI state (keep existing)
    initialize_ui_state()
    
    # Enhanced header (keep existing)
    show_enhanced_header(user_info)
    
    # Track user session (keep existing) 
    track_user_session_start()
    
    # Initialize RAG system if available (keep existing)
    if 'rag_system' not in st.session_state:
        if RAG_SYSTEM_AVAILABLE:
            with st.spinner("Initializing Lumen Navigator AI System..."):
                st.session_state.rag_system = initialize_rag_system()
                
                # Apply enhancements if available
##                if st.session_state.rag_system and PHASE2_AVAILABLE:
##                    st.session_state.rag_system = integrate_phase2_with_existing_rag(st.session_state.rag_system)
                
##                if st.session_state.rag_system and PHASE1_AVAILABLE:
##                    st.session_state.rag_system = enhance_your_rag_system(st.session_state.rag_system)
        else:
            st.session_state.rag_system = None
    
    # Enhanced sidebar (keep existing)
    show_enhanced_sidebar()
    
    # === MODIFIED: Check for beta admin panel ===
    if st.session_state.get('show_beta_admin', False):
        show_beta_admin_panel()
        
        if st.button("← Back to Main App"):
            st.session_state.show_beta_admin = False
            st.rerun()
        
        return  # Don't show main app content when beta admin is active
    
    # Check if analytics dashboard should be shown (keep existing)
    if st.session_state.get('show_analytics', False):
        show_analytics_dashboard(st.session_state.performance_tracker)
        
        if st.button("← Back to Main App"):
            st.session_state.show_analytics = False
            st.rerun()
        
        return  # Don't show main app content when analytics is active
    
    # Track current page view (keep existing)
    if st.session_state.ui_state.get('current_result'):
        track_page_view("results_page")
    elif st.session_state.get('rag_system'):
        track_page_view("full_interface")
    else:
        track_page_view("welcome_interface")
    
    # Main content logic with enhanced styling (keep existing)
    if st.session_state.ui_state.get('current_result'):
        show_enhanced_result_display()
    elif st.session_state.get('rag_system'):
        show_enhanced_full_interface()
    else:
        show_enhanced_welcome_interface()
    
    st.markdown("""
    <div class="app-footer">
        <div style="max-width: 800px; margin: 0 auto;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">🏠</div>
                <div>
                    <strong>Lumen Navigator</strong><br>
                    <span style="color: #9ca3af;">Professional Children's Home Management System</span>
                </div>
            </div>
            <div style="border-top: 1px solid #e5e7eb; padding-top: 1rem; margin-top: 1rem;">
                Powered by advanced AI • Streamlined Auth0 authentication • Lumen Way Homes CIC
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def process_enhanced_request(question, uploaded_files=None, uploaded_images=None):
    """Process user request with enhanced UI feedback and performance tracking"""
    
    # NEW: Track beta query usage
    user_info = st.session_state.get('user_info', {})
    track_beta_query_usage(user_info)
    
    # Get tracker instance
    tracker = st.session_state.performance_tracker
    
    # Updated tracking with image support
    total_attachments = (len(uploaded_files) if uploaded_files else 0) + (len(uploaded_images) if uploaded_images else 0)
    
    # Track feature usage
    tracker.log_feature_usage(
        feature_name="query_submission",
        action="submit_query",
        context={
            "has_attachments": total_attachments > 0,
            "attachment_count": total_attachments,
            "document_count": len(uploaded_files) if uploaded_files else 0,
            "image_count": len(uploaded_images) if uploaded_images else 0,
            "query_length": len(question),
            "performance_mode": st.session_state.get('performance_mode', 'balanced')
        }
    )
    
    try:
        # Enhanced loading state
        progress_placeholder = st.empty()
        
        with progress_placeholder:
            st.markdown("""
            <div class="feature-card loading">
                <div style="text-align: center;">
                    <h4>AI Analysis in Progress</h4>
                    <p>Processing your question and analyzing uploaded content...</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Generating expert guidance..."):
            # Start performance timing
            start_time = time.time()
            
            time.sleep(1)  # Brief pause for UX
            
            # Use RAG system if available
            if st.session_state.get('rag_system'):
                result = st.session_state.rag_system.query(
                    question=question,
                    k=5,
                    response_style="standard",
                    performance_mode=st.session_state.get('performance_mode', 'balanced'),
                    uploaded_files=uploaded_files,
                    uploaded_images=uploaded_images
                )
                
                response_time = time.time() - start_time
                
                query_id = tracker.log_query(
                    query_text=question,
                    response_text=result.get("answer", ""),
                    response_time=response_time,
                    sources=result.get("sources", []),
                    attachments={"documents": uploaded_files, "images": uploaded_images},
                    performance_mode=st.session_state.get('performance_mode', 'balanced'),
                    error_info=None
                )
                
            else:
                # Fallback response
                response_time = time.time() - start_time
                result = {
                    "answer": f"Based on your question: '{question}' - This is a fallback response.",
                    "sources": [],
                    "metadata": {"note": "Fallback response"}
                }
        
        progress_placeholder.empty()
        
        if result and result.get("answer"):
            st.session_state.ui_state.update({
                'current_result': result,
                'current_question': question,
                'show_sources': False
            })
            st.success("Expert guidance generated successfully!")
            st.rerun()
        else:
            st.error("Sorry, I couldn't generate a response. Please try rephrasing your question.")
            
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
            

if __name__ == "__main__":
    main()
