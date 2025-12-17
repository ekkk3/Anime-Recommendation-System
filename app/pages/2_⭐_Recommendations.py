import streamlit as st
from utils import display_anime_card, display_anime_grid
import pandas as pd

anime_df = st.session_state.anime_df
models = st.session_state.models

st.markdown("### ⭐ Nhận Khuyến Nghị Anime")

# Show current model info
current_model = st.session_state.selected_model

if st.session_state.is_admin:
    model_names = {
        'content': '🎯 Dựa Trên Nội Dung',
        'collaborative': '👥 Lọc Hợp Tác', 
        'hybrid': '🔀 Kết Hợp'
    }
    st.info(f"🎛️ Đang sử dụng mô hình **{model_names.get(current_model, current_model.upper())}**")

# ============================================================================
# RECOMMENDATION INTERFACE
# ============================================================================

# For Collaborative Filtering, we need user_id instead of anime_id
if current_model == 'collaborative':
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <strong>👥 Chế Độ Lọc Hợp Tác</strong><br>
        <span style='font-size: 0.9rem;'>Mô hình này cung cấp khuyến nghị cá nhân hóa dựa trên lịch sử đánh giá của người dùng.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Get sample users (users with most ratings)
    ratings_df = st.session_state.ratings_df
    top_users = ratings_df['user_id'].value_counts().head(100).index.tolist()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.selectbox(
            "Chọn User ID (Top 100 người dùng hoạt động nhất):",
            top_users,
            help="Đây là những người dùng thực có lịch sử đánh giá nhiều nhất"
        )
    
    with col2:
        custom_user_id = st.number_input(
            "Hoặc nhập User ID tùy chỉnh:",
            min_value=0,
            value=int(top_users[0]),
            help="Nhập bất kỳ User ID nào từ dataset"
        )
    
    # Use custom if different from selected
    if custom_user_id != user_id:
        user_id = custom_user_id
    
    # Show user stats
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if not user_ratings.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Số Anime Đã Đánh Giá", len(user_ratings))
        with col2:
            st.metric("⭐ Điểm Trung Bình", f"{user_ratings['rating'].mean():.2f}")
        with col3:
            st.metric("🔥 Điểm Cao Nhất", int(user_ratings['rating'].max()))
    
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Số lượng khuyến nghị", 5, 20, 10)
    with col2:
        min_score = st.slider("Điểm anime tối thiểu", 0.0, 10.0, 6.0, 0.5)
    
    if st.button("🎯 Nhận Khuyến Nghị", type="primary", use_container_width=True):
        cf_model = models.get('collaborative')
        
        if cf_model is None:
            st.error("❌ Mô hình Lọc Hợp Tác chưa sẵn sàng")
        else:
            with st.spinner("🔮 Đang tìm khuyến nghị cá nhân hóa..."):
                try:
                    recs = cf_model.recommend_for_user(
                        user_id=user_id,
                        top_n=top_n,
                        min_score=min_score
                    )
                    
                    if len(recs) > 0:
                        st.success(f"✨ Tìm thấy {len(recs)} khuyến nghị cho User {user_id}!")
                        
                        # Show user's watch history first
                        with st.expander("📜 Xem Lịch Sử Đánh Giá Của Người Dùng", expanded=False):
                            user_history = user_ratings.merge(
                                anime_df, 
                                left_on='anime_id', 
                                right_on='MAL_ID'
                            ).sort_values('rating', ascending=False).head(10)
                            
                            st.dataframe(
                                user_history[['Name', 'rating', 'Genres', 'Score', 'Type']],
                                column_config={
                                    "Name": "Tên Anime",
                                    "rating": st.column_config.NumberColumn("Đánh Giá Của Người Dùng", format="%d ⭐"),
                                    "Score": st.column_config.NumberColumn("Điểm Toàn Cầu", format="%.2f"),
                                },
                                use_container_width=True,
                                height=300
                            )
                        
                        st.markdown("---")
                        st.markdown("### 🎬 Khuyến Nghị Cho Bạn")
                        
                        # Display recommendations
                        for _, anime in recs.iterrows():
                            with st.container():
                                st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                                display_anime_card(anime, show_scores=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("😔 Không tìm thấy khuyến nghị. Hãy thử giảm điểm tối thiểu.")
                
                except Exception as e:
                    st.error(f"❌ Lỗi khi tạo khuyến nghị: {str(e)}")
                    st.info("💡 Người dùng này có thể chưa có đủ lịch sử đánh giá. Hãy thử người dùng khác.")

# ============================================================================
# For Content-Based and Hybrid (anime-based recommendations)
# ============================================================================
else:
    if current_model == 'content':
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <strong>🎯 Chế Độ Dựa Trên Nội Dung</strong><br>
            <span style='font-size: 0.9rem;'>Tìm anime tương tự dựa trên thể loại, loại hình và các đặc trưng khác.</span>
        </div>
        """, unsafe_allow_html=True)
    else:  # hybrid
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <strong>🔀 Chế Độ Kết Hợp</strong><br>
            <span style='font-size: 0.9rem;'>Kết hợp ưu điểm cả hai: Tương tự nội dung + Lọc hợp tác.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Anime selection
    selected_anime = st.selectbox(
        "Chọn một anime bạn thích:",
        anime_df['Name'].tolist(),
        help="Chọn một anime làm cơ sở để tìm khuyến nghị"
    )
    
    # Show selected anime info
    selected_info = anime_df[anime_df['Name'] == selected_anime].iloc[0]
    
    with st.expander("📖 Thông Tin Anime Đã Chọn", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⭐ Điểm", f"{selected_info['Score']:.2f}/10")
        with col2:
            st.metric("📺 Loại", selected_info['Type'])
        with col3:
            st.metric("🎬 Số Tập", int(selected_info['Episodes']) if pd.notna(selected_info['Episodes']) else 'N/A')
        with col4:
            st.metric("👥 Thành Viên", f"{int(selected_info['Members']):,}" if pd.notna(selected_info['Members']) else 'N/A')
        
        if pd.notna(selected_info['Genres']):
            genres = selected_info['Genres'].split(',')
            st.markdown("**🎭 Thể Loại:** " + " ".join([f"`{g.strip()}`" for g in genres]))
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Số lượng khuyến nghị", 5, 20, 10)
    with col2:
        min_score = st.slider("Điểm tối thiểu", 0.0, 10.0, 6.0, 0.5)
    
    if st.button("🎯 Nhận Khuyến Nghị", type="primary", use_container_width=True):
        anime_id = anime_df[anime_df['Name'] == selected_anime]['MAL_ID'].values[0]
        model = models.get(current_model)
        
        if model is None:
            st.error(f"❌ Mô hình {current_model.capitalize()} chưa sẵn sàng")
        else:
            with st.spinner("🔍 Đang tìm anime tương tự..."):
                try:
                    # Get recommendations based on model type
                    if current_model == 'hybrid':
                        recs = model.recommend(
                            anime_id=anime_id,
                            top_n=top_n,
                            min_score=min_score
                        )
                    else:  # content-based
                        recs = model.recommend(
                            anime_id,
                            top_n=top_n,
                            min_score=min_score
                        )
                    
                    if len(recs) > 0:
                        st.success(f"✨ Tìm thấy {len(recs)} khuyến nghị!")
                        
                        st.markdown("---")
                        st.markdown("### 🎬 Anime Khuyến Nghị")
                        
                        # Display recommendations
                        for _, anime in recs.iterrows():
                            with st.container():
                                st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                                display_anime_card(anime, show_scores=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("😔 Không tìm thấy khuyến nghị. Hãy thử giảm điểm tối thiểu.")
                
                except Exception as e:
                    st.error(f"❌ Lỗi khi tạo khuyến nghị: {str(e)}")
                    st.info("💡 Hãy thử chọn anime khác hoặc điều chỉnh tham số.")

# ============================================================================
# TIPS SECTION
# ============================================================================
st.markdown("---")
st.markdown("### 💡 Mẹo Để Có Khuyến Nghị Tốt Hơn")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Dựa Trên Nội Dung**
    - Tốt nhất để tìm anime tương tự
    - Hoạt động với mọi anime
    - Dựa trên thể loại & đặc trưng
    - Không cần lịch sử người dùng
    """)

with col2:
    st.markdown("""
    **👥 Lọc Hợp Tác**
    - Cá nhân hóa theo người dùng
    - Khám phá những viên ngọc ẩn
    - Cần lịch sử đánh giá
    - Tốt nhất cho người dùng tích cực
    """)

with col3:
    st.markdown("""
    **🔀 Kết Hợp**
    - Độ chính xác tổng thể tốt nhất
    - Khuyến nghị cân bằng
    - Kết hợp cả hai cách tiếp cận
    - Trọng số linh hoạt (Admin)
    """)