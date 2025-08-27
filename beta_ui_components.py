# beta_ui_components.py - UI Components for Beta Access System

import streamlit as st
from datetime import datetime, timedelta
from beta_access_system import BetaAccessManager
import os

def show_beta_access_gate(user_info):
    """
    Beta access gate - shows after authentication, before main app
    Returns: (access_granted: bool, access_info: dict)
    """
    # Check if user_info is None (authentication not complete)
    if user_info is None:
        st.error("Authentication error: Unable to get user information")
        return False, {"access_granted": False, "status": "auth_error", "message": "Authentication failed"}
    
    if 'beta_manager' not in st.session_state:
        st.session_state.beta_manager = BetaAccessManager()
    
    beta_manager = st.session_state.beta_manager
    access_info = beta_manager.check_user_access(user_info)
    
    if access_info['access_granted']:
        # User has access - show status and proceed
        show_beta_status_sidebar(access_info)
        return True, access_info
    else:
        # User doesn't have access - show appropriate blocking page
        show_beta_blocking_page(access_info)
        return False, access_info

def show_beta_status_sidebar(access_info):
    """Show beta status information in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        
        if access_info['user_type'] == 'admin':
            st.markdown("""
            <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0369a1;">
                <strong style="color: #0369a1;">🛡️ Administrator Access</strong><br>
                <span style="color: #6b7280; font-size: 0.9rem;">Unlimited queries</span>
            </div>
            """, unsafe_allow_html=True)
        
        elif access_info['user_type'] == 'beta_user':
            st.markdown("### 🧪 Beta Access")
            
            # Query usage display
            queries_remaining = access_info['queries_remaining']
            queries_used = access_info['queries_used_today']
            daily_limit = access_info['daily_limit']
            
            # Progress bar calculation
            progress = queries_used / daily_limit if daily_limit > 0 else 0
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600;">Daily Queries</span>
                    <span style="color: #6b7280;">{queries_used}/{daily_limit}</span>
                </div>
                <div style="width: 100%; background: #f3f4f6; border-radius: 4px; height: 8px;">
                    <div style="width: {progress*100}%; background: linear-gradient(90deg, #10b981, #059669); height: 8px; border-radius: 4px;"></div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #6b7280;">
                    {queries_remaining} queries remaining
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Beta user info
            if access_info.get('is_new_user'):
                st.success("✅ Welcome to the beta!")
            
            total_queries = access_info.get('total_queries', 0)
            st.caption(f"Total queries used: {total_queries}")

def show_beta_blocking_page(access_info):
    """Show blocking page for users without access"""
    status = access_info['status']
    
    # Apply the same styling as the main app
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;">
        <div style="font-family: 'Montserrat', sans-serif; font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem;">🏠 Lumen Navigator</div>
        <div style="font-size: 1.2rem; opacity: 0.95;">Beta Testing Program</div>
    </div>
    """, unsafe_allow_html=True)
    
    if status == 'waiting':
        show_waiting_list_page(access_info)
    elif status == 'limit_reached':
        show_query_limit_page(access_info)
    else:
        show_generic_access_denied_page(access_info)

def show_waiting_list_page(access_info):
    """Show waiting list page"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); margin: 1rem 0;">
            <h2 style="color: #1f2937; margin-bottom: 1rem;">📋 You're on the Waiting List!</h2>
            <div style="background: #f3f4f6; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background: #3b82f6; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 1rem;">
                        ⏳
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #1f2937;">Waiting List Status</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Active - awaiting beta spot</div>
                    </div>
                </div>
            </div>
            <p style="color: #6b7280; line-height: 1.6;">
                Thank you for your interest in Lumen Navigator! We're currently at capacity with our beta testing program, 
                but you've been added to our waiting list. We'll notify you as soon as a spot becomes available.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # What happens next
        st.markdown("""
        <div style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 1.5rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
            <h4 style="color: #1e40af; margin-bottom: 1rem;">What happens next?</h4>
            <ul style="color: #1e40af; margin: 0; padding-left: 1.5rem;">
                <li style="margin: 0.5rem 0;">We'll automatically notify you when a beta spot opens</li>
                <li style="margin: 0.5rem 0;">No action needed from you - just wait for our email</li>
                <li style="margin: 0.5rem 0;">Beta spots open as current testers complete their evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Status card
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: #1f2937; margin-bottom: 1rem;">📊 Beta Program Status</h4>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Current Capacity</div>
                <div style="font-weight: 600; color: #dc2626;">Full (20/20 users)</div>
            </div>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Waiting List</div>
                <div style="font-weight: 600; color: #1f2937;">Active</div>
            </div>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Status</div>
                <div style="font-weight: 600; color: #f59e0b;">Awaiting Spot</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact info
        st.markdown("""
        <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <div style="color: #92400e; font-weight: 600; margin-bottom: 0.5rem;">Questions?</div>
            <div style="color: #92400e; font-size: 0.9rem;">
                Email: <a href="mailto:info@lumenwayhomes.org.uk" style="color: #92400e;">info@lumenwayhomes.org.uk</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Refresh button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Check Status Again", type="primary", use_container_width=True):
            st.rerun()

def show_query_limit_page(access_info):
    """Show query limit reached page"""
    reset_time = access_info.get('reset_time', 'midnight')
    queries_used = access_info.get('queries_used_today', 0)
    daily_limit = access_info.get('daily_limit', 15)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
            <h2 style="color: #1f2937; margin-bottom: 1rem;">⏱️ Daily Query Limit Reached</h2>
            <div style="background: #fef3c7; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background: #f59e0b; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 1rem;">
                        {queries_used}
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #92400e;">Queries Used Today</div>
                        <div style="color: #92400e; font-size: 0.9rem;">out of {daily_limit} daily limit</div>
                    </div>
                </div>
            </div>
            <p style="color: #6b7280; line-height: 1.6;">
                You've reached your daily query limit for beta testing. This helps us manage system resources 
                and ensure all beta testers get fair access. Your queries will reset at {reset_time}.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset information
        st.markdown("""
        <div style="background: #dcfce7; border-left: 4px solid #10b981; padding: 1.5rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
            <h4 style="color: #166534; margin-bottom: 1rem;">When do queries reset?</h4>
            <ul style="color: #166534; margin: 0; padding-left: 1.5rem;">
                <li style="margin: 0.5rem 0;">Daily limits reset at midnight (00:00)</li>
                <li style="margin: 0.5rem 0;">You'll get a fresh set of 15 queries</li>
                <li style="margin: 0.5rem 0;">No queries are carried over from previous days</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Usage stats
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: #1f2937; margin-bottom: 1rem;">📈 Usage Statistics</h4>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Today's Usage</div>
                <div style="font-weight: 600; color: #dc2626;">{queries_used}/{daily_limit} queries</div>
            </div>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Resets At</div>
                <div style="font-weight: 600; color: #1f2937;">{reset_time}</div>
            </div>
            <div style="margin: 1rem 0;">
                <div style="color: #6b7280; font-size: 0.9rem;">Status</div>
                <div style="font-weight: 600; color: #f59e0b;">Beta Tester</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips
        st.markdown("""
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <div style="color: #0369a1; font-weight: 600; margin-bottom: 0.5rem;">💡 Beta Testing Tips</div>
            <div style="color: #0369a1; font-size: 0.9rem; line-height: 1.4;">
                • Make the most of your 15 daily queries<br>
                • Try different types of questions<br>
                • Provide feedback on responses
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Refresh and feedback buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("💬 Leave Feedback", use_container_width=True):
            st.info("Feedback feature - contact your administrator for setup")

def show_generic_access_denied_page(access_info):
    """Show generic access denied page"""
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); text-align: center;">
        <h2 style="color: #1f2937; margin-bottom: 1rem;">🚫 Access Not Available</h2>
        <p style="color: #6b7280; line-height: 1.6;">
            We're unable to grant access to Lumen Navigator at this time. 
            Please contact your administrator if you believe this is an error.
        </p>
        <div style="margin-top: 2rem;">
            <a href="mailto:info@lumenwayhomes.org.uk" style="
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                padding: 1rem 2rem;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
            ">Contact Support</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_beta_admin_panel():
    """Show admin panel for beta management"""
    if 'beta_manager' not in st.session_state:
        st.session_state.beta_manager = BetaAccessManager()
    
    beta_manager = st.session_state.beta_manager
    dashboard_data = beta_manager.get_admin_dashboard_data()
    
    st.markdown("# Beta Testing Administration")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Approved Users",
            value=f"{dashboard_data['beta_users']['approved']}/{dashboard_data['beta_users']['limit']}",
            delta=f"{dashboard_data['beta_users']['spots_remaining']} spots remaining"
        )
    
    with col2:
        st.metric(
            label="Waiting List",
            value=dashboard_data['beta_users']['waiting'],
            delta="users waiting"
        )
    
    with col3:
        st.metric(
            label="Active Today",
            value=dashboard_data['daily_stats']['active_users'],
            delta=f"{dashboard_data['daily_stats']['total_queries']} queries"
        )
    
    with col4:
        st.metric(
            label="Avg Queries/User",
            value=f"{dashboard_data['daily_stats']['avg_queries_per_user']:.1f}",
            delta=f"Max: {dashboard_data['usage_trends']['max_daily_queries']}"
        )
    
    st.markdown("---")
    
    # Management actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Promote Users from Waiting List")
        promote_count = st.number_input(
            "Number of users to promote:",
            min_value=0,
            max_value=dashboard_data['beta_users']['spots_remaining'],
            value=min(1, dashboard_data['beta_users']['spots_remaining']),
            help="Promote users from waiting list to approved status"
        )
        
        if st.button("Promote Users", type="primary", disabled=promote_count == 0):
            if promote_count > 0:
                promoted = beta_manager.promote_waiting_list_users(promote_count)
                if promoted:
                    st.success(f"Successfully promoted {len(promoted)} users!")
                    for user in promoted:
                        st.info(f"Promoted: {user['email']} (was position #{user['former_position']})")
                    st.rerun()
                else:
                    st.warning("No users available to promote")
    
    with col2:
        st.markdown("### Usage Statistics")
        st.markdown(f"""
        **Daily Query Limit:** {dashboard_data['usage_trends']['daily_limit']} per user
        
        **7-Day Trends:**
        - Average daily queries: {dashboard_data['usage_trends']['avg_daily_queries']}
        - Peak daily usage: {dashboard_data['usage_trends']['max_daily_queries']}
        """)
    
    # Recent signups table
    st.markdown("---")
    st.markdown("### Recent Signups")
    
    if dashboard_data['recent_signups']:
        signup_data = []
        for signup in dashboard_data['recent_signups']:
            signup_data.append({
                'Email': signup['email'],
                'Name': signup['name'],
                'Status': signup['status'].title(),
                'Position': signup['position'] if signup['position'] else 'N/A',
                'Signup Date': signup['signup_date'][:10]  # Just date part
            })
        
        st.dataframe(signup_data, use_container_width=True)
    else:
        st.info("No recent signups to display")

def track_beta_query_usage(user_info):
    """Track query usage when user makes a request"""
    if 'beta_manager' not in st.session_state:
        st.session_state.beta_manager = BetaAccessManager()
    
    # Don't track admin queries
    email = user_info.get('email', '')
    admin_emails = [
        'garybrooks0@gmail.com', 
        'gbrooks@lumenwayhomes.org.uk',
        'analytics@lumenwayhomes.org.uk'
    ]
    
    is_admin = (email.endswith('@lumenwayhomes.org.uk') or 
               email in admin_emails or 
               os.getenv('LUMEN_ADMIN_MODE') == 'true')
    
    if not is_admin:
        return st.session_state.beta_manager.log_query_usage(user_info)
    
    return True
