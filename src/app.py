"""Main Streamlit application for AppSageAI v2."""

import streamlit as st
import yaml
import os
from logger import logger
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time

from dotenv import load_dotenv

# Import custom modules
from database import PrivacyDatabase
from langfuse_integration import PrivacyLangfuseHandler
from resume_processor import ResumeProcessor

# Load environment variables
load_dotenv()

# Load configuration
def load_config():
    """Load configuration from YAML files."""
    config = {}
    
    # Load main config
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config['app'] = yaml.safe_load(f)
    else:
        # Default config if file doesn't exist
        config['app'] = {
            'app': {
                'name': 'AppSageAI',
                'version': '2.0.0',
                'icon': '🧑‍💻'
            },
            'model': {
                'name': 'llama-3.3-70b-versatile'
            },
            'langfuse': {
                'enabled': True,
                'trace_pii': False
            }
        }
    
    # Load prompts
    prompts_path = Path("config/prompts.yaml")
    if prompts_path.exists():
        with open(prompts_path, 'r') as f:
            config['prompts'] = yaml.safe_load(f)
    else:
        # Default prompts if file doesn't exist
        config['prompts'] = {
            'prompts': {
                'system': {
                    'base': 'You are AppSageAI, an AI assistant for resume analysis. {context}'
                },
                'analysis': {
                    'resume_review': {
                        'title': 'Resume Review',
                        'prompt': 'Analyze the resume against the job description. {context}'
                    }
                }
            }
        }
    
    return config

# Initialize services
@st.cache_resource
def init_services():
    """Initialize all services."""
    services = {}
    
    # Load configuration
    config = load_config()
    services['config'] = config
    
    # Initialize database
    services['db'] = PrivacyDatabase(
        db_path="appsageai.db",
        encryption_key=os.getenv("DB_ENCRYPTION_KEY", "default_dev_key_change_in_prod")
    )
    
    # Initialize Langfuse if enabled
    if config['app'].get('langfuse', {}).get('enabled', False):
        services['langfuse'] = PrivacyLangfuseHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            trace_pii=config['app']['langfuse'].get('trace_pii', False)
        )
    else:
        services['langfuse'] = None
    
    # Initialize resume processor
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
    
    services['processor'] = ResumeProcessor(
        model_name=config['app'].get('model', {}).get('name', 'llama-3.3-70b-versatile'),
        embedding_model=config['app'].get('embeddings', {}).get('model', 'all-MiniLM-L6-v2'),
        chunk_size=config['app'].get('vectorstore', {}).get('chunk_size', 1000),
        chunk_overlap=config['app'].get('vectorstore', {}).get('chunk_overlap', 200)
    )
    
    return services

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = {}
    if 'current_trace_id' not in st.session_state:
        st.session_state.current_trace_id = None

# Feedback component
def render_feedback(response_id: str, services: Dict[str, Any]):
    """Render feedback buttons for a response."""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"up_{response_id}", help="Good response"):
            if response_id not in st.session_state.feedback_given:
                st.session_state.feedback_given[response_id] = "thumbs_up"
                
                # Log to database
                services['db'].store_feedback(
                    st.session_state.session_id,
                    response_id,
                    "thumbs_up"
                )
                
                # Log to Langfuse if available
                if services['langfuse'] and st.session_state.current_trace_id:
                    services['langfuse'].log_feedback(
                        trace_id=st.session_state.current_trace_id,
                        observation_id=response_id,
                        feedback_type="thumbs_up"
                    )
                
                st.success("Thanks for your feedback!")
    
    with col2:
        if st.button("👎", key=f"down_{response_id}", help="Needs improvement"):
            if response_id not in st.session_state.feedback_given:
                st.session_state.feedback_given[response_id] = "thumbs_down"
                
                # Log to database
                services['db'].store_feedback(
                    st.session_state.session_id,
                    response_id,
                    "thumbs_down"
                )
                
                # Log to Langfuse if available
                if services['langfuse'] and st.session_state.current_trace_id:
                    services['langfuse'].log_feedback(
                        trace_id=st.session_state.current_trace_id,
                        observation_id=response_id,
                        feedback_type="thumbs_down"
                    )
                
                st.info("Thanks! We'll work on improving.")

# Main app
def main():
    """Main application function."""
    # Initialize
    services = init_services()
    init_session_state()
    config = services['config']
    
    # Page configuration
    st.set_page_config(
        page_title=config['app']['app']['name'],
        page_icon=config['app']['app']['icon'],
        layout="wide"
    )
    
    # Header
    st.title(f"{config['app']['app']['icon']} {config['app']['app']['name']}")
    st.subheader(config['app']['app']['description'])
    
    # Add privacy notice
    with st.expander("🔒 Privacy & Security"):
        st.info("""
        **Your privacy is our priority:**
        - ✅ All data is encrypted at rest
        - ✅ Personal information is anonymized in logs
        - ✅ Temporary files are deleted immediately
        - ✅ Data is retained for only 30 days
        - ✅ No personally identifiable information is sent to monitoring services
        """)
    
    # Sidebar for user info
    with st.sidebar:
        st.header("👤 User Information")
        
        # User name input
        name = st.text_input("Your Name:", value=st.session_state.user_name or "")
        
        if name and name != st.session_state.user_name:
            st.session_state.user_name = name
            
            # Create new session
            st.session_state.session_id = services['db'].create_session(name)
            
            # Load user memories
            memories = services['db'].get_user_memories(name, limit=5)
            if memories:
                st.success(f"Welcome back, {name}! I remember you.")
                with st.expander("📝 Your previous sessions"):
                    for mem in memories:
                        st.write(f"- {mem['content'].get('summary', 'Previous session')}")
        
        # Job description input
        jd = st.text_area(
            "Job Description:",
            height=200,
            placeholder="Paste the job description here..."
        )
        
        # File upload
        st.header("📄 Resume Upload")
        uploaded_file = st.file_uploader(
            "Upload your Resume",
            type="pdf",
            accept_multiple_files=False,
            help="Maximum file size: 10MB"
        )
        
        # Process resume
        if uploaded_file and name and jd:
            if not st.session_state.resume_processed:
                with st.spinner("Processing your resume..."):
                    # Process the resume
                    success, message, resume_hash = services['processor'].process_resume(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                    
                    if success:
                        st.session_state.resume_processed = True
                        st.success(message)
                        
                        # Store in memory
                        services['db'].store_memory(
                            name,
                            "resume_upload",
                            {
                                "filename": uploaded_file.name,
                                "hash": resume_hash,
                                "job_description": jd[:200],
                                "upload_time": datetime.now().isoformat()
                            }
                        )
                    else:
                        st.error(message)
    
    # Main content area
    if not name:
        st.warning("👈 Please enter your name to continue")
    elif not jd:
        st.warning("👈 Please provide a job description")
    elif not uploaded_file:
        st.warning("👈 Please upload your resume")
    elif st.session_state.resume_processed:
        # Analysis options
        st.header("🔍 Analysis Options")
        
        # Get prompts configuration
        prompts = config['prompts']['prompts']['analysis']
        
        # Create tabs for different analyses
        tabs = st.tabs([
            prompts['resume_review']['title'],
            prompts['skill_improvement']['title'],
            prompts['keyword_analysis']['title'],
            prompts['percentage_match']['title'],
            prompts['cover_letter']['title'],
            "💬 Custom Question"
        ])
        
        # Analysis buttons and results
        analysis_configs = [
            ('resume_review', "Tell Me About the Resume"),
            ('skill_improvement', "How Can I Improve my Skills?"),
            ('keyword_analysis', "What Keywords are Missing?"),
            ('percentage_match', "Calculate Match Percentage"),
            ('cover_letter', "Generate Cover Letter"),
        ]
        
        for idx, (config_key, query) in enumerate(analysis_configs):
            with tabs[idx]:
                if st.button(f"🚀 {prompts[config_key]['title']}", key=f"btn_{config_key}"):
                    with st.spinner(f"Analyzing..."):
                        # Create Langfuse config if available
                        langfuse_config = {}
                        if services['langfuse']:
                            langfuse_config = services['langfuse'].create_trace_config(
                                session_id=st.session_state.session_id,
                                user_id=name,
                                metadata={'analysis_type': config_key}
                            )
                            st.session_state.current_trace_id = langfuse_config.get('run_id')
                        
                        # Perform analysis
                        start_time = time.time()
                        
                        answer, metadata = services['processor'].analyze_resume(
                            prompts[config_key]['prompt'],
                            query,
                            name,
                            jd,
                            langfuse_config
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        # Store conversation
                        conv_id = services['db'].store_conversation(
                            st.session_state.session_id,
                            config_key,
                            {
                                'query': query,
                                'response': answer[:500],  # Store partial for privacy
                                'metadata': metadata
                            }
                        )
                        
                        # Log analytics
                        services['db'].log_analytics(
                            st.session_state.session_id,
                            config_key,
                            int(response_time),
                            metadata.get('tokens_used', 0),
                            metadata.get('model', 'unknown')
                        )
                        
                        # Display results
                        st.success(f"✨ Response time: {response_time:.2f}ms")
                        st.markdown("---")
                        st.markdown(answer)
                        
                        # Render feedback buttons
                        render_feedback(str(conv_id), services)
        
        # Custom question tab
        with tabs[-1]:
            user_question = st.text_input(
                "Ask anything about your resume:",
                placeholder="e.g., How does my experience align with the role?"
            )
            
            if st.button("🔮 Get Answer", key="custom_btn") and user_question:
                with st.spinner("Thinking..."):
                    # Create Langfuse config if available
                    langfuse_config = {}
                    if services['langfuse']:
                        langfuse_config = services['langfuse'].create_trace_config(
                            session_id=st.session_state.session_id,
                            user_id=name,
                            metadata={'analysis_type': 'custom_query'}
                        )
                        st.session_state.current_trace_id = langfuse_config.get('run_id')
                    
                    # Perform analysis
                    start_time = time.time()
                    
                    answer, metadata = services['processor'].analyze_resume(
                        prompts['custom_query']['prompt'],
                        user_question,
                        name,
                        jd,
                        langfuse_config
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    # Store conversation
                    conv_id = services['db'].store_conversation(
                        st.session_state.session_id,
                        'custom_query',
                        {
                            'query': user_question,
                            'response': answer[:500],
                            'metadata': metadata
                        }
                    )
                    
                    # Log analytics
                    services['db'].log_analytics(
                        st.session_state.session_id,
                        'custom_query',
                        int(response_time),
                        metadata.get('tokens_used', 0),
                        metadata.get('model', 'unknown')
                    )
                    
                    # Display results
                    st.success(f"✨ Response time: {response_time:.2f}ms")
                    st.markdown("---")
                    st.markdown(answer)
                    
                    # Render feedback buttons
                    render_feedback(str(conv_id), services)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"Made with ❤️ by AppSageAI Team | v{config['app']['app']['version']} | "
        f"🔒 Privacy-First Design",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()