# beta_access_system.py - Beta Testing Access Control for Lumen Navigator

import sqlite3
import requests
import streamlit as st
import time
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, Tuple, List
import json
import os

class BetaAccessManager:
    """
    Manages beta testing access control with Auth0 integration
    """
    
    def __init__(self):
        self.db_path = "lumen_beta.db"
        self.beta_limit = 20  # Maximum beta users
        self.daily_query_limit = 10  # Queries per day for beta users
        
        # Auth0 Management API configuration
        self.management_domain = self._get_config('AUTH0_DOMAIN')
        self.management_client_id = self._get_config('AUTH0_MANAGEMENT_CLIENT_ID')
        self.management_client_secret = self._get_config('AUTH0_MANAGEMENT_CLIENT_SECRET')
        
        # Initialize database
        self._init_database()

    def get_role_based_limit(self, user_role: str) -> int:
        """Get daily query limit based on user role"""
        ROLE_LIMITS = {
            'manager': 25,
            'safeguarding': 30,
            'inspector': 20,
            'standard': 15
        }
        return ROLE_LIMITS.get(user_role, 15)
    
    def determine_user_role(self, user_info: Dict[str, Any]) -> str:
        """Determine user role from Auth0 app_metadata only"""
        # Get role from Auth0 app_metadata
        app_metadata = user_info.get('app_metadata', {})
        user_role = app_metadata.get('user_role')
        
        if user_role:
            return user_role
        
        # If no role found, return standard as fallback
        # (This should rarely happen with the role collection gate in place)
        return 'standard'

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
    
    def _init_database(self):
        """Initialize SQLite database with all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Beta users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS beta_users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                status TEXT DEFAULT 'waiting',
                signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_date TIMESTAMP,
                waiting_list_position INTEGER,
                total_queries INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Daily usage tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                user_id TEXT,
                usage_date DATE,
                query_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, usage_date),
                FOREIGN KEY (user_id) REFERENCES beta_users (user_id)
            )
        """)
        
        # Waiting list table for additional tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS waiting_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                email TEXT,
                name TEXT,
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notified BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES beta_users (user_id)
            )
        """)
        
        # Emergency overrides table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emergency_overrides (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                override_date DATE,
                justification TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_id TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_management_token(self) -> Optional[str]:
        """Get Auth0 Management API token"""
        if not all([self.management_client_id, self.management_client_secret, self.management_domain]):
            st.warning("⚠️ Auth0 Management API not configured - using database-only mode")
            return None
        
        try:
            # Check if we have a cached token that's still valid
            if hasattr(st.session_state, 'mgmt_token') and hasattr(st.session_state, 'mgmt_token_expires'):
                if time.time() < st.session_state.mgmt_token_expires:
                    return st.session_state.mgmt_token
            
            # Get new token
            token_url = f"https://{self.management_domain}/oauth/token"
            payload = {
                'client_id': self.management_client_id,
                'client_secret': self.management_client_secret,
                'audience': f"https://{self.management_domain}/api/v2/",
                'grant_type': 'client_credentials'
            }
            
            response = requests.post(token_url, json=payload)
            if response.status_code == 200:
                token_data = response.json()
                # Cache token (expires in 1 hour, cache for 50 minutes)
                st.session_state.mgmt_token = token_data['access_token']
                st.session_state.mgmt_token_expires = time.time() + 3000  # 50 minutes
                return token_data['access_token']
            else:
                st.error(f"Failed to get Management API token: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Management API error: {e}")
            return None
    
    def _update_user_metadata(self, user_id: str, metadata: Dict) -> bool:
        """Update user metadata in Auth0"""
        token = self._get_management_token()
        if not token:
            return False
        
        try:
            url = f"https://{self.management_domain}/api/v2/users/{user_id}"
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            # Update app_metadata (not user_metadata for security)
            payload = {
                'app_metadata': metadata
            }
            
            response = requests.patch(url, json=payload, headers=headers)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Auth0 metadata update error: {e}")
            return False
    
    def register_user(self, user_info: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Register user for beta access
        Returns: (status, info_dict)
        """
        user_id = user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name', email)
        
        if not user_id or not email:
            return "error", {"message": "Invalid user information"}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT status, waiting_list_position FROM beta_users WHERE user_id = ?", (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                status, position = existing
                if status == "approved":
                    return "approved", {"message": "Already approved for beta access"}
                else:
                    return "waiting", {"position": position, "message": f"You're #{position} on the waiting list"}
            
            # Check current beta user count
            cursor.execute("SELECT COUNT(*) FROM beta_users WHERE status = 'approved'")
            approved_count = cursor.fetchone()[0]
            
            if approved_count < self.beta_limit:
                # Determine role-based limit for new user
                user_role = self.determine_user_role(user_info)
                role_based_limit = self.get_role_based_limit(user_role)
                
                # Approve immediately - first come, first serve
                cursor.execute("""
                    INSERT INTO beta_users (user_id, email, name, status, approved_date)
                    VALUES (?, ?, ?, 'approved', CURRENT_TIMESTAMP)
                """, (user_id, email, name))
                
                # Update Auth0 metadata
                self._update_user_metadata(user_id, {
                    'beta_status': 'approved',
                    'user_role': user_role,
                    'daily_query_limit': role_based_limit,
                    'approved_date': datetime.now().isoformat()
                })
                
                conn.commit()
                return "approved", {
                    "message": f"Welcome to the beta! You have {role_based_limit} queries per day as a {user_role} user.",
                    "beta_number": approved_count + 1
                }
            
            else:
                # Add to waiting list
                cursor.execute("SELECT MAX(waiting_list_position) FROM beta_users WHERE status = 'waiting'")
                max_position = cursor.fetchone()[0] or 0
                new_position = max_position + 1
                
                cursor.execute("""
                    INSERT INTO beta_users (user_id, email, name, status, waiting_list_position)
                    VALUES (?, ?, ?, 'waiting', ?)
                """, (user_id, email, name, new_position))
                
                # Also add to waiting list tracking table
                cursor.execute("""
                    INSERT INTO waiting_list (user_id, email, name)
                    VALUES (?, ?, ?)
                """, (user_id, email, name))
                
                # Update Auth0 metadata
                self._update_user_metadata(user_id, {
                    'beta_status': 'waiting',
                    'waiting_position': new_position,
                    'signup_date': datetime.now().isoformat()
                })
                
                conn.commit()
                return "waiting", {
                    "position": new_position,
                    "message": f"You're #{new_position} on the waiting list. We'll notify you when a spot opens!",
                    "estimated_spots_ahead": new_position
                }
    
    def check_user_access(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check user's beta access status and daily usage
        Returns comprehensive status information
        """
        user_id = user_info.get('sub')
        email = user_info.get('email', '')
        
        # Check if user is admin
        admin_emails = [
            'garybrooks0@gmail.com', 
            'gbrooks@lumenwayhomes.org.uk',
            'analytics@lumenwayhomes.org.uk'
        ]
        
        is_admin = (email.endswith('@lumenwayhomes.org.uk') or 
                   email in admin_emails or 
                   os.getenv('LUMEN_ADMIN_MODE') == 'true')
        
        if is_admin:
            return {
                'access_granted': True,
                'user_type': 'admin',
                'status': 'admin',
                'message': 'Administrator access - unlimited queries',
                'queries_remaining': 'unlimited',
                'queries_used_today': 0
            }
        
        if not user_id:
            return {
                'access_granted': False,
                'status': 'error',
                'message': 'Unable to identify user'
            }

        # Determine user role and get role-based limit
        user_role = self.determine_user_role(user_info)
        role_based_limit = self.get_role_based_limit(user_role)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get user status
            cursor.execute("""
                SELECT status, waiting_list_position, total_queries, approved_date 
                FROM beta_users WHERE user_id = ?
            """, (user_id,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                # New user - register them
                register_result = self.register_user(user_info)
                status, info = register_result
                
                if status == "approved":
                    return {
                        'access_granted': True,
                        'user_type': 'beta_user', 
                        'status': 'approved',
                        'message': info['message'],
                        'queries_remaining': role_based_limit,
                        'queries_used_today': 0,
                        'beta_number': info['beta_number'],
                        'user_role': user_role,
                        'daily_limit': role_based_limit,
                        'is_new_user': True
                    }
                else:
                    return {
                        'access_granted': False,
                        'user_type': 'waiting',
                        'status': 'waiting',
                        'message': info['message'],
                        'waiting_position': info['position'],
                        'is_new_user': True
                    }
            
            db_status, waiting_position, total_queries, approved_date = user_data
            
            if db_status != 'approved':
                return {
                    'access_granted': False,
                    'user_type': 'waiting',
                    'status': 'waiting',
                    'message': f"You're #{waiting_position} on the waiting list",
                    'waiting_position': waiting_position
                }
            
            # Check daily usage for approved users
            today = date.today()
            cursor.execute("""
                SELECT query_count FROM daily_usage 
                WHERE user_id = ? AND usage_date = ?
            """, (user_id, today))
            
            usage_data = cursor.fetchone()
            queries_used_today = usage_data[0] if usage_data else 0
            queries_remaining = max(0, role_based_limit - queries_used_today)


            if queries_remaining <= 0:
                # Calculate reset time
                tomorrow = datetime.combine(today + timedelta(days=1), datetime.min.time())
                hours_until_reset = (tomorrow - datetime.now()).seconds // 3600
                
                return {
                    'access_granted': False,
                    'user_type': 'beta_user',
                    'status': 'limit_reached',
                    'message': f"Daily limit reached. Resets in {hours_until_reset} hours.",
                    'queries_remaining': 0,
                    'queries_used_today': queries_used_today,
                    'reset_time': tomorrow.strftime('%H:%M'),
                    'daily_limit': role_based_limit,
                    'user_role': user_role,
                    'emergency_override_available': True
                }
            
            return {
                'access_granted': True,
                'user_type': 'beta_user',
                'status': 'approved',
                'message': f"Beta access active - {queries_remaining} queries remaining today",
                'queries_remaining': queries_remaining,
                'queries_used_today': queries_used_today,
                'daily_limit': role_based_limit,
                'user_role': user_role,
                'total_queries': total_queries,
                'approved_date': approved_date,
                'emergency_override_available': queries_remaining <= 2
            }
    
    def log_query_usage(self, user_info: Dict[str, Any]) -> bool:
        """Log a query usage for the user"""
        user_id = user_info.get('sub')
        if not user_id:
            return False
        
        today = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update daily usage
            cursor.execute("""
                INSERT OR REPLACE INTO daily_usage (user_id, usage_date, query_count)
                VALUES (?, ?, COALESCE((SELECT query_count FROM daily_usage WHERE user_id = ? AND usage_date = ?), 0) + 1)
            """, (user_id, today, user_id, today))
            
            # Update total queries
            cursor.execute("""
                UPDATE beta_users SET total_queries = total_queries + 1 WHERE user_id = ?
            """, (user_id,))
            
            conn.commit()
            return True
    
    def get_admin_dashboard_data(self) -> Dict[str, Any]:
        """Get admin dashboard data for beta management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM beta_users WHERE status = 'approved'")
            approved_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM beta_users WHERE status = 'waiting'")
            waiting_count = cursor.fetchone()[0]
            
            # Today's usage
            today = date.today()
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id), SUM(query_count) 
                FROM daily_usage WHERE usage_date = ?
            """, (today,))
            daily_data = cursor.fetchone()
            active_today = daily_data[0] or 0
            queries_today = daily_data[1] or 0
            
            # Recent signups
            cursor.execute("""
                SELECT email, name, signup_date, status, waiting_list_position
                FROM beta_users 
                ORDER BY signup_date DESC 
                LIMIT 10
            """, ())
            recent_signups = cursor.fetchall()
            
            # Usage stats
            cursor.execute("""
                SELECT AVG(query_count), MAX(query_count)
                FROM daily_usage 
                WHERE usage_date >= date('now', '-7 days')
            """, ())
            usage_stats = cursor.fetchone()
            avg_daily = usage_stats[0] or 0
            max_daily = usage_stats[1] or 0
            
            return {
                'beta_users': {
                    'approved': approved_count,
                    'waiting': waiting_count,
                    'limit': self.beta_limit,
                    'spots_remaining': max(0, self.beta_limit - approved_count)
                },
                'daily_stats': {
                    'active_users': active_today,
                    'total_queries': queries_today,
                    'avg_queries_per_user': queries_today / max(active_today, 1)
                },
                'usage_trends': {
                    'avg_daily_queries': round(avg_daily, 1),
                    'max_daily_queries': max_daily,
                    'daily_limit': self.daily_query_limit
                },
                'recent_signups': [
                    {
                        'email': row[0],
                        'name': row[1],
                        'signup_date': row[2],
                        'status': row[3],
                        'position': row[4]
                    }
                    for row in recent_signups
                ]
            }
    
    def promote_waiting_list_users(self, count: int = 1) -> List[Dict]:
        """Promote users from waiting list to approved status"""
        promoted_users = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current approved count
            cursor.execute("SELECT COUNT(*) FROM beta_users WHERE status = 'approved'")
            current_approved = cursor.fetchone()[0]
            
            # Calculate how many we can promote
            available_spots = max(0, self.beta_limit - current_approved)
            promote_count = min(count, available_spots)
            
            if promote_count <= 0:
                return promoted_users
            
            # Get users to promote (lowest waiting list positions)
            cursor.execute("""
                SELECT user_id, email, name, waiting_list_position
                FROM beta_users 
                WHERE status = 'waiting'
                ORDER BY waiting_list_position ASC
                LIMIT ?
            """, (promote_count,))
            
            users_to_promote = cursor.fetchall()
            
            for user_id, email, name, position in users_to_promote:
                # Update status
                cursor.execute("""
                    UPDATE beta_users 
                    SET status = 'approved', approved_date = CURRENT_TIMESTAMP, waiting_list_position = NULL
                    WHERE user_id = ?
                """, (user_id,))
                
                # Update Auth0 metadata
                self._update_user_metadata(user_id, {
                    'beta_status': 'approved',
                    'daily_query_limit': self.daily_query_limit,
                    'promoted_date': datetime.now().isoformat()
                })
                
                promoted_users.append({
                    'user_id': user_id,
                    'email': email,
                    'name': name,
                    'former_position': position
                })
            
            # Update waiting list positions for remaining users
            cursor.execute("""
                UPDATE beta_users 
                SET waiting_list_position = waiting_list_position - ?
                WHERE status = 'waiting' AND waiting_list_position > 0
            """, (promote_count,))
            
            conn.commit()
        
        return promoted_users

    def use_emergency_override(self, user_info: Dict[str, Any], justification: str, query_id: str = None) -> bool:
        """Use emergency override with justification logging"""
        user_id = user_info.get('sub')
        if not user_id:
            return False
        
        # Generate session ID (works in both Streamlit and testing environments)
        try:
            if hasattr(self, 'generate_session_id'):
                session_id = self.generate_session_id()
            else:
                import uuid
                session_id = str(uuid.uuid4())
        except:
            import uuid
            session_id = str(uuid.uuid4())
        today = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Ensure emergency_overrides table exists
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_overrides (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                override_date DATE,
                justification TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_id TEXT
            )
            ''')
            
            # Check if already used 3 overrides today
            cursor.execute("""
                SELECT COUNT(*) FROM emergency_overrides 
                WHERE user_id = ? AND override_date = ?
            """, (user_id, today))
            
            overrides_used = cursor.fetchone()[0]
            
            if overrides_used >= 3:
                return False
            
            # Log the override
            override_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO emergency_overrides (id, user_id, session_id, override_date, justification, query_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (override_id, user_id, session_id, today, justification, query_id))
            
            conn.commit()
            return True
    
    def check_emergency_override_available(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if emergency override is available"""
        user_id = user_info.get('sub')
        if not user_id:
            return {'available': False, 'remaining': 0}
        
        today = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Ensure emergency_overrides table exists
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_overrides (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                override_date DATE,
                justification TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_id TEXT
            )
            ''')
            
            cursor.execute("""
                SELECT COUNT(*) FROM emergency_overrides 
                WHERE user_id = ? AND override_date = ?
            """, (user_id, today))
            
            overrides_used = cursor.fetchone()[0]
            remaining = max(0, 3 - overrides_used)
            
            return {
                'available': remaining > 0,
                'remaining': remaining,
                'used_today': overrides_used
            }
