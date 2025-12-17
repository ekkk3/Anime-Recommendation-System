# app/components/__init__.py
"""
Reusable Streamlit components
"""

from .anime_card import display_anime_card, display_anime_grid
from .sidebar import create_sidebar, display_stats
from .charts import create_score_chart, create_genre_chart, create_type_chart

__all__ = [
    'display_anime_card',
    'display_anime_grid',
    'create_sidebar',
    'display_stats',
    'create_score_chart',
    'create_genre_chart',
    'create_type_chart'
]


# ============================================================================