# streamlined_auth0_fixed.py - Fixed version with proper Streamlit rendering

import streamlit as st
import requests
import time
import urllib.parse
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any
import secrets
import hashlib

class StreamlitAuth0:
    """
    Fixed Auth0 integration with proper Streamlit rendering
    """
    
    def __init__(self):
        # Auth0 configuration from environment or Streamlit secrets
        self.domain = self._get_config('AUTH0_DOMAIN')
        self.client_id = self._get_config('AUTH0_CLIENT_ID')
        self.client_secret = self._get_config('AUTH0_CLIENT_SECRET')
        self.redirect_uri = self._get_config('AUTH0_REDIRECT_URI', self._get_default_redirect_uri())

        if not all([self.domain, self.client_id, self.client_secret]):
            st.error("❌ Auth0 configuration missing. Please check your secrets.")
            st.stop()

    def _get_default_redirect_uri(self) -> str:
        """Auto-detect redirect URI based on environment"""
        # Check if running on Streamlit Cloud
        if 'streamlit.app' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
            return 'https://lumen-navigator-app.streamlit.app'
        # Default to localhost for development
        return 'http://localhost:8501'

    
    def _get_config(self, key: str, default: str = None) -> str:
        """Get configuration from environment or Streamlit secrets"""
        # Try environment variables first
        value = os.getenv(key)
        if value:
            return value
        
        # Try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
            
            # Try nested auth0 section
            auth0_secrets = st.secrets.get('auth0', {})
            if key in auth0_secrets:
                return auth0_secrets[key]
                
        except Exception:
            pass
        
        return default
    
    def _apply_login_styles(self):
        """Apply CSS styles for login page using Streamlit's proper method"""
        st.markdown("""
        <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@600;700&display=swap');
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .css-1rs6os.edgvbvh3 {visibility: hidden;}
        
        /* Main container styling */
        .main .block-container {
            padding: 1rem;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Hero header */
        .login-hero {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .hero-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            opacity: 0.95;
            margin-bottom: 0.5rem;
        }
        
        .hero-description {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        /* Cards */
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border-left: 4px solid #f59e0b;
            margin: 1rem 0;
        }
        
        .security-card {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Login button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _get_logo_component(self):
        """Return logo as a Streamlit component"""
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <svg viewBox="0 0 200 160" style="width: 120px; height: 96px;">
                <defs>
                    <radialGradient id="lightGrad" cx="50%" cy="30%">
                        <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:0.3"/>
                        <stop offset="100%" style="stop-color:#4682B4;stop-opacity:0.1"/>
                    </radialGradient>
                    <linearGradient id="houseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#5DADE2"/>
                        <stop offset="50%" style="stop-color:#3498DB"/>
                        <stop offset="100%" style="stop-color:#2E86C1"/>
                    </linearGradient>
                    <linearGradient id="pathGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#85C1E9"/>
                        <stop offset="50%" style="stop-color:#5DADE2"/>
                        <stop offset="100%" style="stop-color:#3498DB"/>
                    </linearGradient>
                </defs>
                
                <circle cx="100" cy="50" r="80" fill="url(#lightGrad)"/>
                <line x1="100" y1="20" x2="100" y2="5" stroke="#87CEEB" stroke-width="2"/>
                <line x1="130" y1="30" x2="140" y2="20" stroke="#87CEEB" stroke-width="1.5"/>
                <line x1="70" y1="30" x2="60" y2="20" stroke="#87CEEB" stroke-width="1.5"/>
                <line x1="145" y1="50" x2="160" y2="50" stroke="#87CEEB" stroke-width="1.5"/>
                <line x1="55" y1="50" x2="40" y2="50" stroke="#87CEEB" stroke-width="1.5"/>
                
                <polygon points="100,35 85,50 115,50" fill="url(#houseGrad)"/>
                <rect x="88" y="50" width="24" height="18" fill="url(#houseGrad)"/>
                <rect x="94" y="56" width="5" height="5" fill="white" opacity="0.8"/>
                <rect x="101" y="56" width="5" height="5" fill="white" opacity="0.8"/>
                
                <path d="M 100 70 Q 80 90 60 110 Q 40 130 70 140 Q 100 150 130 140 Q 160 130 140 110 Q 120 90 100 70"
                      fill="none" stroke="url(#pathGrad)" stroke-width="8" opacity="0.7"/>
                <path d="M 100 70 Q 85 85 70 100 Q 55 115 75 125 Q 95 135 115 125 Q 135 115 120 100 Q 105 85 100 70"
                      fill="none" stroke="white" stroke-width="4" opacity="0.5"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
    
    def login_button(self, text: str = "🔐 Sign in with Auth0") -> None:
        """Display login button that redirects to Auth0"""
        # Generate and store state parameter
        state = self._generate_and_store_state()
        
        # Build Auth0 authorization URL
        auth_url = self._build_auth_url(state)
        
        # Use Streamlit's link_button if available (newer versions)
        try:
            if hasattr(st, 'link_button'):
                st.link_button(text, auth_url, use_container_width=True)
            else:
                # Fallback for older Streamlit versions
                st.markdown(f"""
                <a href="{auth_url}" target="_self" style="
                    display: block;
                    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                    color: white;
                    padding: 1rem 2rem;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 600;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    {text}
                </a>
                """, unsafe_allow_html=True)
        except Exception:
            # Final fallback
            st.markdown(f"[{text}]({auth_url})")
    
    def _generate_and_store_state(self) -> str:
        """Generate state parameter and store it"""
        state = secrets.token_urlsafe(16)
        st.session_state['auth0_state'] = state
        st.session_state['auth0_timestamp'] = time.time()
        return state
    
    def _build_auth_url(self, state: str) -> str:
        """Build Auth0 authorization URL"""
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid profile email',
            'state': state
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"https://{self.domain}/authorize?{query_string}"
    
    def handle_callback(self) -> bool:
        """Handle Auth0 callback"""
        query_params = st.query_params
        
        if 'code' not in query_params:
            return False
        
        code = query_params.get('code')
        state = query_params.get('state')
        error = query_params.get('error')
        
        if error:
            st.error(f"❌ Authentication failed: {error}")
            return False
        
        if not state:
            st.error("❌ No state parameter received")
            return False
        
        try:
            tokens = self._exchange_code_for_tokens(code)
            if not tokens:
                return False
            
            user_info = self._get_user_info(tokens['access_token'])
            if not user_info:
                return False
            
            self._store_user_session(user_info, tokens)
            st.query_params.clear()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Authentication failed: {str(e)}")
            return False
    
    def _exchange_code_for_tokens(self, code: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for tokens"""
        token_url = f"https://{self.domain}/oauth/token"
        
        payload = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(token_url, data=payload, headers=headers)
        
        if response.status_code != 200:
            st.error(f"❌ Token exchange failed: {response.text}")
            return None
        
        return response.json()
    
    def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from Auth0"""
        userinfo_url = f"https://{self.domain}/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}
        
        response = requests.get(userinfo_url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"❌ Failed to get user info: {response.text}")
            return None
        
        return response.json()
    
    def _store_user_session(self, user_info: Dict[str, Any], tokens: Dict[str, Any]):
        """Store user session in Streamlit session state"""
        st.session_state.update({
            'authenticated': True,
            'user_info': user_info,
            'access_token': tokens.get('access_token'),
            'id_token': tokens.get('id_token'),
            'session_start': time.time(),
            'auth_provider': 'auth0'
        })
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        if not st.session_state.get('authenticated', False):
            return False
        
        # Check session timeout (2 hours)
        session_start = st.session_state.get('session_start', 0)
        if time.time() - session_start > 7200:  # 2 hours
            self.logout()
            return False
        
        return True
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        if self.is_authenticated():
            return st.session_state.get('user_info')
        return None
    
    def logout(self):
        """Logout user and clear session"""
        # Clear session state
        auth_keys = [
            'auth0_state', 'authenticated', 'user_info', 'access_token', 
            'id_token', 'session_start', 'auth_provider'
        ]
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Redirect to Auth0 logout
        logout_url = f"https://{self.domain}/v2/logout?client_id={self.client_id}&returnTo={self.redirect_uri}"
        st.markdown(f'<meta http-equiv="refresh" content="0; url={logout_url}">', unsafe_allow_html=True)
    
    def show_user_info(self):
        """Display user information in sidebar"""
        if not self.is_authenticated():
            return
        
        user_info = self.get_user_info()
        if not user_info:
            return
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Session")
            
            name = user_info.get('name', user_info.get('nickname', 'Professional'))
            email = user_info.get('email', 'No email')
            picture = user_info.get('picture')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if picture:
                    st.image(picture, width=50)
                else:
                    st.markdown(f"""
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #3b82f6, #2563eb); 
                         border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                         color: white; font-weight: bold; font-size: 1.2rem;">
                        {name[0] if name else 'U'}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.write(f"**{name}**")
                st.caption(f"📧 {email}")
            
            st.success("🔐 Authenticated via Auth0")
            
            # Session timer
            session_start = st.session_state.get('session_start', time.time())
            elapsed = time.time() - session_start
            remaining = max(0, 7200 - elapsed)  # 2 hours
            
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            
            st.write(f"**Session:** {hours}h {minutes}m remaining")
            
            if st.button("🚪 Logout", use_container_width=True):
                self.logout()
    
    def require_authentication(self) -> bool:
        """Require authentication - main entry point"""
        # Handle callback first
        if self.handle_callback():
            st.success("✅ Successfully authenticated!")
            time.sleep(1)
            st.rerun()
        
        # Check if already authenticated
        if self.is_authenticated():
            self.show_user_info()
            return True
        
        # Show login page
        self._show_login_page()
        return False
    
    def _show_login_page(self):
        """Show the login page with proper Streamlit components"""
        # Apply styles
        self._apply_login_styles()
        
        # Header with gradient
        st.markdown("""
        <div class="login-hero">
            <div class="hero-title">🏠 Lumen Navigator</div>
            <div class="hero-subtitle">Professional Children's Home Management System</div>
            <div class="hero-description">
                Secure, AI-powered guidance for children's residential care professionals
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logo
        # self._get_logo_component()
        
        # Main content in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔐 Secure Access Required")
            st.markdown("""
            Professional authentication is required to access specialized children's home 
            management tools and AI-powered guidance systems.
            """)
            
            # Security info
            st.markdown("""
            <div class="security-card">
                <h4 style="color: #0369a1; margin-bottom: 1rem;">🛡️ Enterprise Security</h4>
                <ul style="color: #0369a1; margin: 0; padding-left: 1rem;">
                    <li>Professional Auth0 authentication</li>
                    <li>Secure session management</li>
                    <li>Multi-factor authentication support</li>
                    <li>Comprehensive audit logging</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Login button
            self.login_button("🔐 Sign in with Auth0")
        
        with col2:
            st.markdown("### 🎯 System Capabilities")
            
            # Feature cards
            features = [
                ("🧠 AI Analysis", "Intelligent guidance and recommendations"),
                ("📊 Ofsted Reports", "Automated analysis and pathways"),
                ("🛡️ Safeguarding", "Framework guidance and assessment"),
                ("📋 Compliance", "Policy checking and documentation")
            ]
            
            for title, desc in features:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">{title}</div>
                    <div style="color: #6b7280; font-size: 0.9rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <script>
            // Check if we're in a popup and authenticated
            if (window.opener && window.name === 'auth0_login') {
                // If this is a popup login window, close it after successful auth
                setTimeout(function() {
                    if (window.location.href.includes('code=')) {
                        window.close();
                    }
                }, 2000);
            }
            </script>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6b7280; padding: 1rem;">
            <strong>Need Access?</strong><br>
            Contact your administrator or email: 
            <a href="mailto:info@lumenwayhomes.org.uk" style="color: #3b82f6;">info@lumenwayhomes.org.uk</a>
        </div>
        """, unsafe_allow_html=True)
