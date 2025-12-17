import streamlit as st
import pandas as pd
import hashlib
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.hybrid_recommender import HybridRecommender
from config import *
from utils import display_anime_card, display_anime_grid

# Page config
st.set_page_config(
    page_title="🎌 Anime Recommender", 
    page_icon="🎌", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations
st.markdown("""<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);
        }
    }
    
    @keyframes modelSwitch {
        0% {
            opacity: 0;
            transform: scale(0.95);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 2rem 0;
        margin-bottom: 1rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .anime-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid transparent;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .anime-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .anime-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .anime-score {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: transform 0.3s ease;
    }
    
    .anime-score:hover {
        transform: scale(1.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.05);
        animation: glow 2s ease-in-out infinite;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }
    
    .genre-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        transition: transform 0.2s ease;
    }
    
    .genre-badge:hover {
        transform: translateY(-2px);
    }
    
    .model-selector {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        animation: slideInRight 0.6s ease-out;
    }
    
    .model-button {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .model-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .model-button:hover::before {
        left: 100%;
    }
    
    .model-button:hover {
        border-color: #667eea;
        transform: translateX(5px) scale(1.02);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .model-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        animation: modelSwitch 0.3s ease-out;
    }
    
    .model-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        animation: modelSwitch 0.4s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .model-indicator::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .model-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        transition: transform 0.3s ease;
    }
    
    .model-badge:hover {
        transform: scale(1.1);
    }
    
    .model-badge.content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .model-badge.collaborative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .model-badge.hybrid {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: slideInRight 0.5s ease-out;
    }
    
    .info-card:hover {
        background: #e9ecef;
        border-left-width: 6px;
        padding-left: 1.2rem;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
</style>""", unsafe_allow_html=True)

# Session state init
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'hybrid'
if 'previous_model' not in st.session_state:
    st.session_state.previous_model = None

# Load functions
@st.cache_data
def load_data():
    try:
        return pd.read_csv(ANIME_CLEANED)
    except:
        st.error("❌ Cannot load data.")
        return None

@st.cache_data
def load_ratings():
    try:
        return pd.read_csv(RATING_SAMPLED, usecols=['user_id', 'anime_id', 'rating'])
    except Exception as e:
        st.error(f"Error loading ratings: {e}")
        return None

@st.cache_resource
def load_all_models():
    models = {}
    with st.spinner('🔄 Loading AI models...'):
        # Content-based
        try:
            cb = ContentBasedRecommender()
            cb.load_model(CONTENT_MODEL, TFIDF_VECTORIZER)
            models['content'] = cb
            time.sleep(0.2)
        except Exception as e:
            models['content'] = None
        
        # Collaborative
        try:
            cf = CollaborativeFilteringRecommender()
            cf.load_model(COLLABORATIVE_MODEL)
            models['collaborative'] = cf
            time.sleep(0.2)
        except Exception as e:
            models['collaborative'] = None
        
        # Hybrid
        try:
            if models['content'] and models['collaborative']:
                hybrid = HybridRecommender(
                    models['content'], 
                    models['collaborative'], 
                    content_weight=0.4, 
                    collaborative_weight=0.6
                )
                models['hybrid'] = hybrid
            else:
                models['hybrid'] = None
        except Exception as e:
            models['hybrid'] = None
    
    return models

# Load data & models
anime_df = load_data()
if anime_df is None:
    st.stop()
ratings_df = load_ratings()

if 'models' not in st.session_state:
    st.session_state.models = load_all_models()

models = st.session_state.models
st.session_state.anime_df = anime_df
st.session_state.ratings_df = ratings_df

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

def get_model_info(model_key):
    """Get model display information with emojis"""
    info = {
        'content': {
            'name': 'Content-Based',
            'icon': '🎯',
            'color': 'content',
            'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'description': 'TF-IDF + Cosine Similarity',
            'speed': '⚡⚡⚡ Ultra Fast',
            'best_for': 'Similar anime discovery',
            'emoji': '🎯'
        },
        'collaborative': {
            'name': 'Collaborative Filtering',
            'icon': '👥',
            'color': 'collaborative',
            'gradient': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'description': 'SVD Matrix Factorization',
            'speed': '⚡⚡ Fast',
            'best_for': 'Personalized recommendations',
            'emoji': '👥'
        },
        'hybrid': {
            'name': 'Hybrid Model',
            'icon': '🔀',
            'color': 'hybrid',
            'gradient': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            'description': 'CB (40%) + CF (60%)',
            'speed': '⚡⚡ Fast',
            'best_for': 'Best overall performance',
            'emoji': '🔀'
        }
    }
    return info.get(model_key, {})

# ============================================================================
# SIDEBAR - Enhanced Model Selector
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/anime.png", width=80)
    st.markdown('<div style="animation: fadeInUp 0.5s ease-out;">', unsafe_allow_html=True)
    st.title("🎯 Anime Recommender")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats with animation
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anime", f"{len(anime_df):,}")
    with col2:
        st.metric("Avg Score", f"{anime_df['Score'].mean():.2f}")
    
    st.markdown("---")
    
    # Admin login/panel
    if not st.session_state.is_admin:
        st.markdown("### 🔐 Admin Login")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin123")
            if st.form_submit_button("🚀 Login", use_container_width=True):
                if username == ADMIN_USERNAME and hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
    else:
        # Admin panel with animated model selector
        st.success("✅ Admin Mode Active")
        
        st.markdown("### 🎛️ Model Configuration")
        
        # Get available models
        available_models = []
        model_map = {}
        
        if models.get('content'):
            available_models.append('content')
            model_map['content'] = '🎯 Content-Based'
        
        if models.get('collaborative'):
            available_models.append('collaborative')
            model_map['collaborative'] = '👥 Collaborative'
        
        if models.get('hybrid'):
            available_models.append('hybrid')
            model_map['hybrid'] = '🔀 Hybrid'
        
        # Display model selection buttons with animation
        for model_key, display_name in model_map.items():
            info = get_model_info(model_key)
            is_selected = st.session_state.selected_model == model_key
            
            # Button with animation on click
            if st.button(
                display_name,
                key=f"model_btn_{model_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                if st.session_state.selected_model != model_key:
                    st.session_state.previous_model = st.session_state.selected_model
                    st.session_state.selected_model = model_key
                    # Show brief loading animation
                    with st.spinner(f'🔄 Switching to {info["name"]}...'):
                        time.sleep(0.3)
                    st.rerun()
            
            # Show info for selected model
            if is_selected:
                st.markdown(f"""
                <div class='info-card' style='background: {info["gradient"]}; color: white; border-left: none;'>
                    <strong>📝 {info['description']}</strong><br>
                    {info['speed']}<br>
                    <em>{info['best_for']}</em>
                </div>
                """, unsafe_allow_html=True)
        
        # Hybrid configuration slider
        if st.session_state.selected_model == 'hybrid' and models.get('hybrid'):
            st.markdown("---")
            st.markdown("#### ⚙️ Hybrid Weights")
            
            cb_weight = st.slider(
                "Content Weight",
                min_value=0.0,
                max_value=1.0,
                value=models['hybrid'].content_weight,
                step=0.1,
                help="Weight for content-based recommendations"
            )
            
            # Update weights with animation feedback
            if cb_weight != models['hybrid'].content_weight:
                models['hybrid'].content_weight = cb_weight
                models['hybrid'].collaborative_weight = 1 - cb_weight
                st.success(f"✅ Updated: CB {cb_weight:.0%} | CF {1-cb_weight:.0%}")
            
            col1, col2 = st.columns(2)
            col1.metric("CB", f"{cb_weight:.1%}")
            col2.metric("CF", f"{1-cb_weight:.1%}")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-header">🎌 Anime Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Anime Discovery Platform</p>', unsafe_allow_html=True)

# Current model indicator with animation
if st.session_state.is_admin:
    current_model_info = get_model_info(st.session_state.selected_model)
    st.markdown(f"""
    <div class='model-indicator'>
        <div style='position: relative; z-index: 1;'>
            <strong style='font-size: 1.5rem;'>{current_model_info['emoji']} Active Model: {current_model_info['name']}</strong>
            <br>
            <span style='font-size: 1rem; opacity: 0.9;'>{current_model_info['description']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Metrics with staggered animation
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card" style="animation-delay: 0s;"><div class="metric-label">📺 TOTAL ANIME</div><div class="metric-value">17K+</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card" style="animation-delay: 0.2s;"><div class="metric-label">⭐ AVG SCORE</div><div class="metric-value">{anime_df["Score"].mean():.1f}</div></div>', unsafe_allow_html=True)
with col3:
    genres_count = len(anime_df['Genres'].str.split(',').explode().unique())
    st.markdown(f'<div class="metric-card" style="animation-delay: 0.4s;"><div class="metric-label">🎭 GENRES</div><div class="metric-value">{genres_count}</div></div>', unsafe_allow_html=True)

st.markdown("### 🎯 Discover Anime")

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    min_score = st.slider("Min Score", 0.0, 10.0, 7.0, 0.5)
with col2:
    types = ["All"] + sorted(anime_df['Type'].unique().tolist())
    anime_type = st.selectbox("Type", types)
with col3:
    sort_by = st.selectbox("Sort by", ["Score", "Members", "Favorites"])
with col4:
    top_n = st.slider("Show", 5, 50, 20, 5)

# Apply filters
filtered = anime_df[anime_df['Score'] >= min_score].copy()
if anime_type != "All":
    filtered = filtered[filtered['Type'] == anime_type]

top_anime = filtered.nlargest(top_n, sort_by)
st.markdown(f"### 🏆 Top {len(top_anime)} Anime")

# Display results with animation
for i in range(0, len(top_anime), 2):
    cols = st.columns(2)
    with cols[0]:
        if i < len(top_anime):
            with st.container():
                st.markdown(f'<div class="anime-card" style="animation-delay: {i*0.1}s;">', unsafe_allow_html=True)
                display_anime_card(top_anime.iloc[i])
                st.markdown('</div>', unsafe_allow_html=True)
    with cols[1]:
        if i + 1 < len(top_anime):
            with st.container():
                st.markdown(f'<div class="anime-card" style="animation-delay: {(i+1)*0.1}s;">', unsafe_allow_html=True)
                display_anime_card(top_anime.iloc[i + 1])
                st.markdown('</div>', unsafe_allow_html=True)