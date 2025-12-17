import streamlit as st
import pandas as pd
from utils import display_anime_grid

anime_df = st.session_state.anime_df
ratings_df = st.session_state.ratings_df
models = st.session_state.models

st.title("📜 Phân Tích Lịch Sử Người Dùng Thực Tế")

# --- KHỞI TẠO STATE CHO HISTORY ---
if 'top_users_hist' not in st.session_state:
    top_users = ratings_df['user_id'].value_counts().head(50).index.tolist()
    st.session_state.top_users_hist = top_users
    st.session_state.selected_user_hist = top_users[0]
    st.session_state.custom_user_id_hist = top_users[0]
    st.session_state.user_id_hist = top_users[0]

top_users = st.session_state.top_users_hist

# Callbacks
def on_select_change():
    new_val = st.session_state.selected_user_hist_key
    st.session_state.selected_user_hist = new_val
    st.session_state.custom_user_id_hist = new_val
    st.session_state.user_id_hist = new_val

def on_custom_change():
    new_val = st.session_state.custom_user_hist_key
    st.session_state.custom_user_id_hist = new_val
    st.session_state.user_id_hist = new_val

# Widget chọn user
try:
    current_index = top_users.index(st.session_state.selected_user_hist)
except ValueError:
    current_index = 0

st.selectbox(
    "Chọn User ID (Top 50 người dùng hoạt động nhất):",
    top_users,
    index=current_index,
    key="selected_user_hist_key",
    on_change=on_select_change
)

st.number_input(
    "Hoặc nhập User ID bất kỳ:",
    min_value=0,
    value=st.session_state.custom_user_id_hist,
    key="custom_user_hist_key",
    on_change=on_custom_change
)

user_id = st.session_state.user_id_hist

# Lịch sử user
user_history = ratings_df[ratings_df['user_id'] == user_id]
if user_history.empty:
    st.warning(f"User ID {user_id} chưa có lịch sử.")
else:
    full_history = user_history.merge(anime_df, left_on='anime_id', right_on='MAL_ID').sort_values('rating', ascending=False)
    
    st.markdown(f"### 👤 Hồ sơ User: `{user_id}`")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Đã xem", len(full_history))
    with c2:
        st.metric("Điểm trung bình", f"{full_history['rating'].mean():.2f}/10")
    with c3:
        try:
            fav_genre = full_history['Genres'].str.split(',').explode().mode()[0].strip()
            st.metric("Genre yêu thích", fav_genre)
        except:
            st.metric("Genre yêu thích", "N/A")
    with c4:
        st.metric("Điểm cao nhất", f"{full_history['rating'].max()}/10")
    
    st.divider()
    st.subheader(f"📺 Lịch sử xem ({len(full_history)} anime)")
    st.dataframe(
        full_history[['Name', 'rating', 'Genres', 'Type', 'Episodes', 'Score']],
        column_config={
            "Name": "Tên Anime",
            "rating": st.column_config.NumberColumn("Đánh Giá Của Người Dùng", format="%d ⭐"),
            "Score": st.column_config.NumberColumn("Điểm Toàn Cầu", format="%.2f"),
        },
        use_container_width=True,
        height=300
    )
    
    st.markdown("#### ⭐ Anime Được Đánh Giá Cao Nhất Bởi Người Dùng Này")
    display_anime_grid(full_history.head(4), columns=4)
    
    st.divider()
    st.subheader(f"🤖 Gợi Ý AI Cho User {user_id}")
    if st.button("🚀 Tạo gợi ý cho User này", type="primary"):
        cf_model = models.get('collaborative')
        if cf_model:
            with st.spinner("Đang phân tích..."):
                try:
                    recs = cf_model.recommend_for_user(user_id, top_n=8)
                    if not recs.empty:
                        st.success("Dựa trên lịch sử, AI nghĩ bạn sẽ thích:")
                        display_anime_grid(recs, columns=4, show_scores=True)
                    else:
                        st.warning("Không tìm thấy gợi ý.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.error("Model Collaborative chưa load.")