import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from heartbeat_sim import DroneHeartbeatSimulator
import time
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(
    page_title="无人机心跳监测系统",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .stAlert {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .offline-warning {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.title("🚁 无人机智能化应用 - 心跳监测系统")
st.markdown("---")

# 侧边栏控制
with st.sidebar:
    st.header("🎮 控制面板")
    
    if 'simulator' not in st.session_state:
        st.session_state.simulator = DroneHeartbeatSimulator()
        st.session_state.running = False
        st.session_state.offline_alarm = False
        
    if not st.session_state.running:
        if st.button("▶️ 启动无人机", use_container_width=True):
            st.session_state.simulator.start()
            st.session_state.running = True
            st.session_state.offline_alarm = False
            st.rerun()
    else:
        if st.button("⏹️ 停止无人机", use_container_width=True):
            st.session_state.simulator.stop()
            st.session_state.running = False
            st.rerun()
    
    st.markdown("---")
    st.info("""
    **系统说明**
    - 📡 心跳频率: 1次/秒
    - ⚠️ 掉线阈值: 3秒无响应
    - 📊 实时可视化心跳数据
    """)

# 主显示区域
col1, col2, col3 = st.columns(3)

# 获取实时数据
if st.session_state.running:
    latest = st.session_state.simulator.get_latest_heartbeat()
    history = st.session_state.simulator.get_history()
    
    # 检查掉线状态
    if st.session_state.simulator.offline and not st.session_state.offline_alarm:
        st.session_state.offline_alarm = True
        st.balloons()  # 掉线时抛气球提示
        
    # 显示指标
    with col1:
        if latest and latest.get('status') == 'alive':
            st.metric("📡 当前状态", "🟢 在线", delta="正常")
        elif latest and latest.get('status') == 'offline':
            st.metric("📡 当前状态", "🔴 掉线", delta=f"已离线 {latest.get('offline_after',0)}秒", delta_color="inverse")
        else:
            st.metric("📡 当前状态", "⚪ 待机", delta="等待连接")
            
    with col2:
        if latest and latest.get('status') == 'alive':
            st.metric("⏱️ 最近心跳", latest.get('time_str', '--'), delta="实时更新")
        elif latest:
            st.metric("⏱️ 最后在线", latest.get('time_str', '--'), delta="已掉线")
        else:
            st.metric("⏱️ 最近心跳", "--", delta="无数据")
            
    with col3:
        heartbeat_count = len([h for h in history if h.get('status') == 'alive'])
        st.metric("💗 累计心跳", f"{heartbeat_count} 次", delta=f"共{len(history)}次传输")
        
    # 报警区域
    if st.session_state.simulator.offline:
        st.error("🚨 **警报！无人机已掉线超过3秒！** 请检查连接！")
        
    # 实时心跳显示
    if latest and latest.get('status') == 'alive':
        with st.expander("📡 当前心跳详情", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**心跳时间:** {latest['time_str']}")
                st.write(f"**心跳序号:** #{latest['heartbeat_id']}")
            with col_b:
                st.write(f"**呼吸时间:** {latest['breath_time']} 秒")
                st.write(f"**信号质量:** {'优秀' if latest['breath_time'] < 1.0 else '良好' if latest['breath_time'] < 1.1 else '正常'}")
else:
    st.info("👈 请点击左侧「启动无人机」开始监测")
    history = []

# 可视化图表
st.markdown("---")
st.subheader("📈 实时心跳趋势图")

if history and len(history) > 0:
    # 准备数据
    df_data = []
    for h in history:
        df_data.append({
            '时间': h['time_str'],
            '呼吸时间(秒)': h['breath_time'] if h.get('status') == 'alive' else None,
            '状态': '🟢在线' if h.get('status') == 'alive' else '🔴掉线',
            '序号': h['heartbeat_id']
        })
    
    df = pd.DataFrame(df_data)
    
    # 使用Plotly创建交互式图表
    fig = go.Figure()
    
    # 添加在线数据点
    online_df = df[df['状态'] == '🟢在线']
    if len(online_df) > 0:
        fig.add_trace(go.Scatter(
            x=online_df['序号'],
            y=online_df['呼吸时间(秒)'],
            mode='lines+markers',
            name='🟢 在线状态',
            line=dict(color='green', width=2),
            marker=dict(size=8, color='green'),
            text=online_df['时间'],
            hovertemplate='序号: %{x}<br>时间: %{text}<br>呼吸时间: %{y}秒<extra></extra>'
        ))
    
    # 添加掉线标记
    offline_df = df[df['状态'] == '🔴掉线']
    if len(offline_df) > 0:
        fig.add_trace(go.Scatter(
            x=offline_df['序号'],
            y=[0] * len(offline_df),
            mode='markers',
            name='🔴 掉线标记',
            marker=dict(size=15, color='red', symbol='x'),
            text=offline_df['时间'],
            hovertemplate='序号: %{x}<br>时间: %{text}<br>⚠️ 无人机掉线<extra></extra>'
        ))
    
    fig.update_layout(
        title="无人机心跳信号实时监控",
        xaxis_title="心跳序号",
        yaxis_title="呼吸时间 (秒)",
        hovermode='closest',
        height=500,
        template='plotly_white'
    )
    
    # 添加水平线表示正常范围
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="理想值 1.0秒")
    fig.add_hrect(y0=0.8, y1=1.2, line_width=0, fillcolor="green", opacity=0.1, annotation_text="正常范围")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 数据表格
    with st.expander("📋 查看详细数据"):
        st.dataframe(df, use_container_width=True)
        
else:
    st.info("启动无人机后，心跳数据将在此处实时显示")

# 实时刷新
if st.session_state.running:
    time.sleep(0.5)
    st.rerun()

# 页脚
st.markdown("---")
st.caption(f"🕒 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 无人机智能化应用系统 v1.0")
