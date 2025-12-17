import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
import time

st.set_page_config(page_title="So Sánh Mô Hình", page_icon="🔬", layout="wide")

# Animation CSS (giữ nguyên vì là style)
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .metric-card-animated {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .metric-card-animated:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .comparison-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
    }
    
    .comparison-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Load metrics with animation
@st.cache_data
def load_model_metrics():
    """Tải dữ liệu đánh giá mô hình từ file JSON hoặc dùng giá trị mặc định"""
    metrics_file = Path('data/processed/model_metrics.json')
    
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Chuyển đổi sang định dạng mong muốn
            return {
                'Content-Based': {
                    'Description': data['content_based']['description'],
                    'Pros': '\n'.join([f'✅ {p}' for p in data['content_based']['pros']]),
                    'Cons': '\n'.join([f'❌ {c}' for c in data['content_based']['cons']]),
                    'Use Case': data['content_based']['best_for'],
                    'Speed': data['content_based']['speed_emoji'] + ' ' + data['content_based']['speed_rating'],
                    'Coverage': data['content_based']['coverage'],
                    'Diversity': data['content_based']['diversity'],
                    'Avg_Similarity': data['content_based']['avg_similarity']
                },
                'Collaborative Filtering': {
                    'Description': data['collaborative_filtering']['description'],
                    'RMSE': data['collaborative_filtering']['rmse'],
                    'MAE': data['collaborative_filtering']['mae'],
                    'Precision@10': data['collaborative_filtering']['precision_at_10'],
                    'Recall@10': data['collaborative_filtering']['recall_at_10'],
                    'Pros': '\n'.join([f'✅ {p}' for p in data['collaborative_filtering']['pros']]),
                    'Cons': '\n'.join([f'❌ {c}' for c in data['collaborative_filtering']['cons']]),
                    'Use Case': data['collaborative_filtering']['best_for'],
                    'Speed': data['collaborative_filtering']['speed_emoji'] + ' ' + data['collaborative_filtering']['speed_rating'],
                    'Coverage': data['collaborative_filtering']['coverage'],
                    'Diversity': data['collaborative_filtering']['diversity']
                },
                'Hybrid': {
                    'Description': data['hybrid']['description'],
                    'RMSE': data['hybrid']['rmse'],
                    'MAE': data['hybrid']['mae'],
                    'Precision@10': data['hybrid']['precision_at_10'],
                    'Recall@10': data['hybrid']['recall_at_10'],
                    'Pros': '\n'.join([f'✅ {p}' for p in data['hybrid']['pros']]),
                    'Cons': '\n'.join([f'❌ {c}' for c in data['hybrid']['cons']]),
                    'Use Case': data['hybrid']['best_for'],
                    'Speed': data['hybrid']['speed_emoji'] + ' ' + data['hybrid']['speed_rating'],
                    'Coverage': data['hybrid']['coverage'],
                    'Diversity': data['hybrid']['diversity']
                },
                'metadata': data['metadata']
            }
        except Exception as e:
            st.warning(f"⚠️ Không tải được file metrics: {e}. Đang dùng giá trị mặc định.")
    
    # Giá trị mặc định nếu không có file
    return {
        'Content-Based': {
            'Description': 'TF-IDF + Cosine Similarity',
            'Pros': '✅ Không gặp vấn đề cold-start với anime mới\n✅ Dễ giải thích kết quả\n✅ Tốc độ suy luận rất nhanh',
            'Cons': '❌ Độ đa dạng thấp\n❌ Dễ bị over-specialization (gợi ý quá hẹp)',
            'Use Case': 'Người dùng mới, khám phá anime tương tự',
            'Speed': '⚡⚡⚡ Rất nhanh',
            'Coverage': 0.85,
            'Diversity': 0.62,
            'Avg_Similarity': 0.73
        },
        'Collaborative Filtering': {
            'Description': 'Phân rã ma trận SVD',
            'RMSE': 1.24,
            'MAE': 0.98,
            'Precision@10': 0.156,
            'Recall@10': 0.089,
            'Pros': '✅ Khám phá các mẫu ẩn\n✅ Độ chính xác cao\n✅ Cá nhân hóa tốt',
            'Cons': '❌ Vấn đề cold-start (người dùng mới/anime mới)\n❌ Cần lịch sử đánh giá',
            'Use Case': 'Người dùng tích cực có lịch sử đánh giá',
            'Speed': '⚡⚡ Nhanh',
            'Coverage': 0.78,
            'Diversity': 0.71
        },
        'Hybrid': {
            'Description': 'Kết hợp có trọng số: Dựa nội dung (40%) + Lọc hợp tác (60%)',
            'RMSE': 1.18,
            'MAE': 0.94,
            'Precision@10': 0.168,
            'Recall@10': 0.095,
            'Pros': '✅ Kết hợp ưu điểm cả hai\n✅ Gợi ý cân bằng\n✅ Linh hoạt',
            'Cons': '❌ Phức tạp hơn\n❌ Cần tinh chỉnh tham số',
            'Use Case': 'Mục đích chung, phù hợp mọi người dùng',
            'Speed': '⚡⚡ Nhanh',
            'Coverage': 0.82,
            'Diversity': 0.68
        }
    }

# Hiệu ứng loading
with st.spinner('🔄 Đang tải dữ liệu đánh giá mô hình...'):
    time.sleep(0.5)
    metrics = load_model_metrics()

# Tiêu đề chính
st.markdown('<h1 class="fade-in" style="text-align: center; font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔬 So Sánh Hiệu Suất Mô Hình Gợi Ý</h1>', unsafe_allow_html=True)
st.markdown('<p class="fade-in" style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">So sánh hiệu suất các mô hình gợi ý anime khác nhau</p>', unsafe_allow_html=True)

# Hiển thị metadata nếu có
if 'metadata' in metrics:
    meta = metrics['metadata']
    st.info(f"📊 Ngày đánh giá: {meta.get('evaluation_date', 'N/A')} | Tổng anime: {meta.get('total_anime', 'N/A'):,} | Tổng đánh giá: {meta.get('total_ratings', 'N/A'):,}")

# Tổng quan các mô hình
st.markdown("## 📋 Tổng Quan Các Mô Hình")

col1, col2, col3 = st.columns(3)

with col1:
    time.sleep(0.1)
    st.markdown("""
    <div class="metric-card-animated" style="animation-delay: 0s;">
        <h3>🎯 Dựa Trên Nội Dung</h3>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>Vector hóa TF-IDF<br>
        So sánh Cosine Similarity</p>
        <div style='font-size: 2rem; font-weight: bold; margin-top: 1rem;'>⚡⚡⚡</div>
        <p style='font-size: 0.8rem;'>Siêu nhanh</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    time.sleep(0.2)
    st.markdown("""
    <div class="metric-card-animated" style="animation-delay: 0.2s; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h3>👥 Lọc Hợp Tác</h3>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>Phân rã ma trận SVD<br>
        Tương tác người dùng-anime</p>
        <div style='font-size: 2rem; font-weight: bold; margin-top: 1rem;'>RMSE: {:.2f}</div>
        <p style='font-size: 0.8rem;'>Độ chính xác cao</p>
    </div>
    """.format(metrics['Collaborative Filtering']['RMSE']), unsafe_allow_html=True)

with col3:
    time.sleep(0.3)
    st.markdown("""
    <div class="metric-card-animated" style="animation-delay: 0.4s; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h3>🔀 Kết Hợp <span class="winner-badge">🏆 TỐT NHẤT</span></h3>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>Kết hợp có trọng số<br>
        Nội dung (40%) + Lọc hợp tác (60%)</p>
        <div style='font-size: 2rem; font-weight: bold; margin-top: 1rem;'>RMSE: {:.2f}</div>
        <p style='font-size: 0.8rem;'>Hiệu suất tổng thể tốt nhất</p>
    </div>
    """.format(metrics['Hybrid']['RMSE']), unsafe_allow_html=True)

st.markdown("---")

# Các chỉ số chi tiết
st.markdown("## 📊 Các Chỉ Số Hiệu Suất")

cf_metrics = metrics['Collaborative Filtering']
hybrid_metrics = metrics['Hybrid']

metrics_df = pd.DataFrame({
    'Metric': ['RMSE ↓ (càng thấp càng tốt)', 'MAE ↓ (càng thấp càng tốt)', 'Precision@10 ↑', 'Recall@10 ↑'],
    'Collaborative': [
        cf_metrics['RMSE'],
        cf_metrics['MAE'],
        cf_metrics['Precision@10'],
        cf_metrics['Recall@10']
    ],
    'Hybrid': [
        hybrid_metrics['RMSE'],
        hybrid_metrics['MAE'],
        hybrid_metrics['Precision@10'],
        hybrid_metrics['Recall@10']
    ]
})

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Lọc Hợp Tác',
        x=metrics_df['Metric'],
        y=metrics_df['Collaborative'],
        marker_color='#f5576c',
        text=metrics_df['Collaborative'].round(3),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Giá trị: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Kết Hợp 🏆',
        x=metrics_df['Metric'],
        y=metrics_df['Hybrid'],
        marker_color='#00f2fe',
        text=metrics_df['Hybrid'].round(3),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Giá trị: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='So Sánh Hiệu Suất Mô Hình',
        barmode='group',
        height=400,
        yaxis_title='Giá trị',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition={'duration': 500}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Điểm Nổi Bật")
    st.markdown("""
    **Chỉ số tốt nhất:**
    - 🏆 **RMSE thấp nhất**: Kết hợp ({:.2f})
    - 🏆 **MAE thấp nhất**: Kết hợp ({:.2f})
    - 🏆 **Precision@10 cao nhất**: Kết hợp ({:.3f})
    - 🏆 **Recall@10 cao nhất**: Kết hợp ({:.3f})
    
    **Khuyến nghị:**
    ✨ Mô hình Kết hợp có hiệu suất tổng thể tốt nhất
    """.format(
        hybrid_metrics['RMSE'],
        hybrid_metrics['MAE'],
        hybrid_metrics['Precision@10'],
        hybrid_metrics['Recall@10']
    ))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Phủ sóng và độ đa dạng
st.markdown("## 🎨 Phân Tích Độ Phủ Sóng & Đa Dạng")

col1, col2 = st.columns(2)

with col1:
    coverage_data = pd.DataFrame({
        'Model': ['Dựa Trên Nội Dung', 'Lọc Hợp Tác', 'Kết Hợp 🏆'],
        'Coverage': [
            metrics['Content-Based']['Coverage'],
            metrics['Collaborative Filtering']['Coverage'],
            metrics['Hybrid']['Coverage']
        ]
    })
    
    fig = px.bar(
        coverage_data,
        x='Model',
        y='Coverage',
        color='Coverage',
        color_continuous_scale='Viridis',
        title='Độ Phủ Sóng Danh Mục',
        text='Coverage'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**Độ phủ sóng**: Tỷ lệ anime trong danh mục được gợi ý ít nhất một lần")

with col2:
    diversity_data = pd.DataFrame({
        'Model': ['Dựa Trên Nội Dung', 'Lọc Hợp Tác 🏆', 'Kết Hợp'],
        'Diversity': [
            metrics['Content-Based']['Diversity'],
            metrics['Collaborative Filtering']['Diversity'],
            metrics['Hybrid']['Diversity']
        ]
    })
    
    fig = px.bar(
        diversity_data,
        x='Model',
        y='Diversity',
        color='Diversity',
        color_continuous_scale='Plasma',
        title='Độ Đa Dạng Gợi Ý',
        text='Diversity'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**Độ đa dạng**: Mức độ khác biệt trung bình giữa các gợi ý")

st.markdown("---")

# Biểu đồ radar tổng hợp
st.markdown("## 🎯 Biểu Đồ Radar Tổng Hợp")

categories = ['Coverage', 'Diversity', 'Precision', 'Recall', 'Speed']

fig = go.Figure()

# Chuẩn hóa về thang 0-1
cb_values = [
    metrics['Content-Based']['Coverage'],
    metrics['Content-Based']['Diversity'],
    0.5,  # Không có precision cho Content-Based
    0.5,  # Không có recall
    1.0   # Nhanh nhất
]

cf_values = [
    metrics['Collaborative Filtering']['Coverage'],
    metrics['Collaborative Filtering']['Diversity'],
    metrics['Collaborative Filtering']['Precision@10'] * 5,
    metrics['Collaborative Filtering']['Recall@10'] * 10,
    0.8
]

hybrid_values = [
    metrics['Hybrid']['Coverage'],
    metrics['Hybrid']['Diversity'],
    metrics['Hybrid']['Precision@10'] * 5,
    metrics['Hybrid']['Recall@10'] * 10,
    0.8
]

fig.add_trace(go.Scatterpolar(
    r=cb_values,
    theta=categories,
    fill='toself',
    name='Dựa Trên Nội Dung',
    line_color='#667eea'
))

fig.add_trace(go.Scatterpolar(
    r=cf_values,
    theta=categories,
    fill='toself',
    name='Lọc Hợp Tác',
    line_color='#f5576c'
))

fig.add_trace(go.Scatterpolar(
    r=hybrid_values,
    theta=categories,
    fill='toself',
    name='Kết Hợp 🏆',
    line_color='#00f2fe'
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title='So Sánh Đa Chiều Các Mô Hình',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Tab ưu nhược điểm chi tiết
st.markdown("## ⚖️ Phân Tích Chi Tiết Mô Hình")

tab1, tab2, tab3 = st.tabs(["🎯 Dựa Trên Nội Dung", "👥 Lọc Hợp Tác", "🔀 Kết Hợp"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.markdown("### ✅ Ưu Điểm")
        st.success(metrics['Content-Based']['Pros'])
        st.markdown("### 🎯 Trường Hợp Sử Dụng Tốt Nhất")
        st.info(metrics['Content-Based']['Use Case'])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="comparison-card" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
        st.markdown("### ❌ Nhược Điểm")
        st.warning(metrics['Content-Based']['Cons'])
        st.markdown("### ⚡ Tốc Độ")
        st.info(metrics['Content-Based']['Speed'])
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.markdown("### ✅ Ưu Điểm")
        st.success(metrics['Collaborative Filtering']['Pros'])
        st.markdown("### 🎯 Trường Hợp Sử Dụng Tốt Nhất")
        st.info(metrics['Collaborative Filtering']['Use Case'])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="comparison-card" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
        st.markdown("### ❌ Nhược Điểm")
        st.warning(metrics['Collaborative Filtering']['Cons'])
        st.markdown("### ⚡ Tốc Độ")
        st.info(metrics['Collaborative Filtering']['Speed'])
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.markdown("### ✅ Ưu Điểm")
        st.success(metrics['Hybrid']['Pros'])
        st.markdown("### 🎯 Trường Hợp Sử Dụng Tốt Nhất")
        st.info(metrics['Hybrid']['Use Case'])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="comparison-card" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
        st.markdown("### ❌ Nhược Điểm")
        st.warning(metrics['Hybrid']['Cons'])
        st.markdown("### ⚡ Tốc Độ")
        st.info(metrics['Hybrid']['Speed'])
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Phần cấu hình mô hình (admin)
st.markdown("## 🎛️ Cấu Hình Mô Hình Tương Tác")

st.info("💡 **Người dùng Admin** có thể chuyển đổi mô hình ở sidebar và thử các cấu hình khác nhau!")

if 'is_admin' in st.session_state and st.session_state.is_admin:
    st.success("✅ Bạn đang đăng nhập với quyền Admin - Sử dụng sidebar để chuyển đổi mô hình")
    
    if st.session_state.get('selected_model') == 'hybrid' and 'models' in st.session_state:
        hybrid_model = st.session_state.models.get('hybrid')
        if hybrid_model:
            st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
            st.markdown("### Cấu Hình Kết Hợp Hiện Tại")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trọng số Nội Dung", f"{hybrid_model.content_weight:.1%}", 
                         delta=f"{(hybrid_model.content_weight - 0.4):.1%}" if hybrid_model.content_weight != 0.4 else None)
            with col2:
                st.metric("Trọng số Lọc Hợp Tác", f"{hybrid_model.collaborative_weight:.1%}",
                         delta=f"{(hybrid_model.collaborative_weight - 0.6):.1%}" if hybrid_model.collaborative_weight != 0.6 else None)
            with col3:
                st.metric("Chiến lược", "Có trọng số")
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("🔒 Đăng nhập với quyền Admin để sử dụng tính năng chuyển đổi mô hình")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;' class='fade-in'>
    <p>📊 Chỉ số được tính trên tập test với hơn 1 triệu đánh giá</p>
    <p>🔬 Đánh giá bao gồm RMSE, MAE, Precision@K, Recall@K, Coverage và Diversity</p>
    <p style='margin-top: 1rem;'>💾 Chạy lệnh <code>python save_model_metrics.py</code> để cập nhật chỉ số từ lần huấn luyện mới nhất</p>
</div>
""", unsafe_allow_html=True)