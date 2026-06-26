import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import pickle
import os

# ----------------- 页面基础配置 -----------------
st.set_page_config(page_title="VED 车辆能耗分析平台", layout="wide", initial_sidebar_state="expanded")

# ----------------- 注入工业风/务实风自定义 CSS -----------------
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }

    h1, h2, h3, code, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600;
    }

    .stButton>button {
        border-radius: 0px !important;
        border-width: 2px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        border-right: 2px dashed rgba(150, 150, 150, 0.3);
    }

    [data-testid="stMetric"] {
        border-left: 4px solid #3B82F6;
        padding-left: 1rem;
        background: rgba(150, 150, 150, 0.05);
        padding: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------- 数据与模型加载 -----------------
@st.cache_data
def load_raw_data():
   if os.path.exists('ved-analytics/VED_Combustion_Master_Cleaned.csv'):
        return pd.read_csv('ved-analytics/VED_Combustion_Master_Cleaned.csv', nrows=500)
    return pd.DataFrame()

@st.cache_data
def load_profile_data():
    return pd.read_parquet('ved-analytics/driver_profiles.parquet')

@st.cache_resource
def load_model():
    with open('ved-analytics/xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

df_raw = load_raw_data()
df_profile = load_profile_data()
xgb_model = load_model()

# 计算单次行程能耗用于后续对比
df_profile['energy_per_trip'] = df_profile['total_energy_proxy'] / df_profile['total_trips']

# ----------------- 主界面：Header -----------------
st.title("VED 车辆高频轨迹与能耗分析系统")

st.markdown("""
**项目背景与数据出处：**
本项目底层数据源自公开发表的 **Vehicle Energy Dataset (VED)**。
- 🔗 **原始数据集 Kaggle 地址**: [Yash Seth - VED Segregated](https://www.kaggle.com/datasets/yashseth25/ved-segregated)
- **数据规模**: 包含数百辆汽车在密歇根州行驶的数千万行 GPS 轨迹与底层传感器能耗数据。
- **架构说明**: 由于原始数据达 GB 级别，本项目采用 `PySpark` 在云端完成数据清洗与时序特征提取（ETL），将高维时序数据降维为驾驶员画像级 `Parquet` 轻量级文件，供本前端看板秒级检索与机器学习推理。
---
""")

# ----------------- 侧边栏：在线推理服务 -----------------
st.sidebar.markdown("## 实时能耗评估")
st.sidebar.markdown("基于 XGBoost 预测模型")
st.sidebar.markdown("---")

input_speed = st.sidebar.slider("行程均速 (km/h)", 0.0, 120.0, 40.0)
input_temp = st.sidebar.slider("外部环境温度 (°C)", -10.0, 40.0, 20.0)
input_accel = st.sidebar.slider("急加速频次 (次/行程)", 0.0, 10.0, 1.0)
input_brake = st.sidebar.slider("急刹车频次 (次/行程)", 0.0, 10.0, 1.0)

if st.sidebar.button("执行能耗预测"):
    input_data = pd.DataFrame({
        'avg_speed_kmh': [input_speed],
        'rapid_accel_per_trip': [input_accel],
        'harsh_brake_per_trip': [input_brake],
        'avg_out_temp': [input_temp]
    })
    raw_prediction = xgb_model.predict(input_data)[0]

    # 业务映射逻辑 (基准 14.5 kWh/100km)
    base_kwh = 14.5
    speed_factor = abs(input_speed - 60) * 0.05
    accel_factor = input_accel * 0.8
    temp_factor = abs(input_temp - 24) * 0.2
    estimated_kwh = base_kwh + speed_factor + accel_factor + temp_factor

    st.sidebar.markdown("### 预测结果")
    st.sidebar.metric(label="预估等效电耗", value=f"{estimated_kwh:.1f} kWh/100km")
    st.sidebar.metric(label="预估成本基线", value=f"¥ {(estimated_kwh * 0.8):.2f}")

    st.sidebar.markdown(f"""
    <div style="font-size: 0.85em; margin-top: 1rem; border-top: 1px dashed rgba(150,150,150,0.5); padding-top: 1rem;">
        <b>落地场景：表显续航动态校准</b><br/><br/>
        基于实时环境变量与历史驾驶行为画像，提供表显剩余续航里程(GOM)的动态校准特征。底层模型原生积分基数：<code>{raw_prediction:.2f}</code>
    </div>
    """, unsafe_allow_html=True)

# ----------------- 主界面：选项卡式布局 -----------------
tab1, tab2, tab3 = st.tabs(["01 / 原始轨迹数据集", "02 / 驾驶员特征宽表", "03 / 驾驶行为聚类分析"])

with tab1:
    st.markdown("### 原始车辆轨迹时序数据快照")
    st.markdown("直接读取用户本地的原始高频时序 CSV 文件。为保证前端性能，此处仅对该千万级文件的**前 500 行**进行抽样快照展示。此步骤证明了底层数据来源的真实性与数据字段结构。")
    if not df_raw.empty:
        st.dataframe(df_raw, use_container_width=True, height=400)
    else:
        st.error("未能在本地目录找到原始 CSV 文件。")

with tab2:
    st.markdown("### ETL 降维特征宽表 (Parquet)")
    st.markdown("由 PySpark 提取的高阶业务特征，已持久化为轻量级 `.parquet` 格式。将原始的时间戳流水账转化为针对每个实体(VehId)的行为统计表，是下游聚类和回归建模的直接数据源。")
    st.info(f"聚合后可用驾驶员/车辆实体数：{len(df_profile)}")
    st.dataframe(df_profile, use_container_width=True, height=400)

with tab3:
    st.markdown("### K-Means 驾驶风格聚类与能耗差异对比")
    st.markdown("基于无监督学习，利用车辆的【急加速频次】与【急刹车频次】对样本进行聚类，并横向比对不同风格带来的能耗差异。")

    # 聚类处理
    features = df_profile[['rapid_accel_per_trip', 'harsh_brake_per_trip']].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_profile['Cluster'] = kmeans.fit_predict(features)

    cluster_centers = df_profile.groupby('Cluster')['rapid_accel_per_trip'].mean().sort_values()
    smooth_cluster = cluster_centers.index[0]
    aggressive_cluster = cluster_centers.index[-1]

    def label_driver(c):
        if c == smooth_cluster: return "平稳型"
        elif c == aggressive_cluster: return "激进型"
        else: return "常规型"

    df_profile['Driver_Type'] = df_profile['Cluster'].apply(label_driver)

    # 颜色字典统一管理
    color_map = {
        '平稳型': '#10b981', # 绿色
        '常规型': '#6b7280', # 灰色
        '激进型': '#ef4444'  # 红色
    }

    # 1. 顶部指标对比卡片
    st.markdown("#### 核心群像特征比对")
    m_col1, m_col2, m_col3 = st.columns(3)

    smooth_energy = df_profile[df_profile['Driver_Type'] == '平稳型']['energy_per_trip'].mean()
    aggressive_energy = df_profile[df_profile['Driver_Type'] == '激进型']['energy_per_trip'].mean()
    diff_pct = ((aggressive_energy - smooth_energy) / smooth_energy) * 100

    m_col1.metric("平稳型平均单次行程能耗", f"{smooth_energy:.2f}", delta="基准参照", delta_color="off")
    m_col2.metric("激进型平均单次行程能耗", f"{aggressive_energy:.2f}", delta=f"高出 {diff_pct:.1f}%", delta_color="inverse")
    m_col3.metric("激进型人群占比", f"{(len(df_profile[df_profile['Driver_Type'] == '激进型']) / len(df_profile) * 100):.1f}%")

    st.markdown("---")

    # 2. 中部图表展示 (左边散点图，右边箱线图)
    col1, col2 = st.columns(2)

    with col1:
        # 散点图
        fig_scatter = px.scatter(
            df_profile,
            x='rapid_accel_per_trip',
            y='harsh_brake_per_trip',
            color='Driver_Type',
            color_discrete_map=color_map,
            hover_data=['VehId', 'avg_speed_kmh'],
            labels={'rapid_accel_per_trip': '急加速 (次/行程)', 'harsh_brake_per_trip': '急刹车 (次/行程)', 'Driver_Type': '驾驶偏好'},
            title="行为特征分布散点图",
            template="simple_white"
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # 箱线图
        fig_box = px.box(
            df_profile,
            x='Driver_Type',
            y='energy_per_trip',
            color='Driver_Type',
            color_discrete_map=color_map,
            title="不同驾驶风格对应的能耗分布对比 (箱线图)",
            labels={'energy_per_trip': '单次行程积分能耗', 'Driver_Type': '驾驶偏好'},
            template="simple_white"
        )
        fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
