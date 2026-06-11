import streamlit as st
import folium
from streamlit_folium import st_folium
from folium import plugins
import math
import json
import os
import random
import time

# ==================== 页面配置 ====================
st.set_page_config(page_title="无人机航线规划系统", page_icon="🛰️", layout="wide")

# ==================== 南京科技职业学院坐标 ====================
CAMPUS = [32.234097, 118.749413]

# ==================== 初始化 ====================
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = []
if 'point_a' not in st.session_state:
    st.session_state.point_a = [32.2323, 118.749]
if 'point_b' not in st.session_state:
    st.session_state.point_b = [32.2344, 118.749]
if 'flight_alt' not in st.session_state:
    st.session_state.flight_alt = 50
if 'route_plans' not in st.session_state:
    st.session_state.route_plans = []
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None
if 'confirmed_plan' not in st.session_state:
    st.session_state.confirmed_plan = None
if 'temp_obs' not in st.session_state:
    st.session_state.temp_obs = None
if 'temp_h' not in st.session_state:
    st.session_state.temp_h = [0, 60]
if 'temp_name' not in st.session_state:
    st.session_state.temp_name = "建筑物"

# ==================== 工具函数 ====================
def calc_distance(p1, p2):
    """计算两点距离（米）"""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    R = 6371000
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def point_in_polygon(point, poly):
    """射线法判断点是否在多边形内"""
    x, y = point[1], point[0]
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][1], poly[i][0]
        x2, y2 = poly[(i+1)%n][1], poly[(i+1)%n][0]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1)+x1):
            inside = not inside
    return inside

class Obstacle:
    """3D障碍物类"""
    def __init__(self, points, min_h, max_h, name):
        self.points = points
        self.min_h = min_h
        self.max_h = max_h
        self.name = name
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.cx = (min(lats)+max(lats))/2
        self.cy = (min(lons)+max(lons))/2
    
    def contains(self, point):
        return point_in_polygon(point, self.points)
    
    def get_bypass_left(self, start, end):
        """获取左侧绕行点"""
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        L = math.sqrt(dx*dx+dy*dy)
        if L > 0:
            dx /= L
            dy /= L
        perp_x = -dy
        perp_y = dx
        offset = 0.004  # 绕行距离
        return [self.cx + perp_y*offset, self.cy + perp_x*offset]
    
    def get_bypass_right(self, start, end):
        """获取右侧绕行点"""
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        L = math.sqrt(dx*dx+dy*dy)
        if L > 0:
            dx /= L
            dy /= L
        perp_x = -dy
        perp_y = dx
        offset = 0.004  # 绕行距离
        return [self.cx - perp_y*offset, self.cy - perp_x*offset]


# ==================== 标题 ====================
st.title("🛰️ 无人机智能航线规划系统")
st.markdown("**南京科技职业学院** | 障碍物高度设置 | 多航线选择（左绕行/右绕行/最佳航线）")
st.markdown("---")

# ==================== 两列布局 ====================
col_left, col_right = st.columns([1.5, 1])

# ==================== 左侧：地图 ====================
with col_left:
    st.subheader("🗺️ 卫星地图")
    
    m = folium.Map(location=CAMPUS, zoom_start=17, control_scale=True)
    folium.TileLayer(
        tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
        attr='高德卫星', subdomains=['1','2','3','4'], name='卫星'
    ).add_to(m)
    
    # 学院标记
    folium.Marker(CAMPUS, popup="🏫 南京科技职业学院", icon=folium.Icon(color='red')).add_to(m)
    
    # 绘制障碍物
    for obs in st.session_state.obstacles:
        if len(obs.points) >= 3:
            # 根据高度关系选择颜色
            if st.session_state.flight_alt < obs.min_h:
                color = 'red'  # 需要绕行
            elif st.session_state.flight_alt > obs.max_h:
                color = 'green'  # 可飞越
            else:
                color = 'orange'  # 冲突
            folium.Polygon(
                obs.points, color=color, weight=2, fill=True, 
                fill_color=color, fill_opacity=0.3,
                popup=f"{obs.name}\n高度: {obs.min_h}-{obs.max_h}m\n飞行高度: {st.session_state.flight_alt}m"
            ).add_to(m)
    
    # A/B点
    folium.Marker(st.session_state.point_a, popup="🚁 起点A", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(st.session_state.point_b, popup="🎯 终点B", icon=folium.Icon(color='red')).add_to(m)
    
    # 绘制选中的航线
    if st.session_state.selected_plan:
        p = st.session_state.selected_plan
        folium.PolyLine(p['points'], color=p['color'], weight=4, opacity=0.9).add_to(m)
        if len(p['points']) > 2:
            folium.Marker(p['points'][1], popup="绕行点", icon=folium.Icon(color='purple')).add_to(m)
    
    # 绘图工具
    plugins.Draw(draw_options={'polygon': {'allowIntersection': False}}).add_to(m)
    plugins.MeasureControl().add_to(m)
    folium.LayerControl().add_to(m)
    
    output = st_folium(m, width=700, height=500, key="map")
    
    if output and output.get('last_active_drawing'):
        d = output['last_active_drawing']
        if d and d['geometry']['type'] == 'Polygon':
            pts = [[c[1], c[0]] for c in d['geometry']['coordinates'][0]]
            if len(pts) >= 3:
                st.session_state.temp_obs = pts
                st.success(f"✅ 已绘制 {len(pts)} 个点")

# ==================== 右侧：控制面板 ====================
with col_right:
    st.subheader("🎯 控制面板")
    
    # ========== 新建障碍物 ==========
    if st.session_state.temp_obs:
        st.markdown("### 🆕 新建3D障碍物")
        st.caption(f"已绘制 {len(st.session_state.temp_obs)} 个边界点")
        
        name = st.text_input("障碍物名称", value=st.session_state.temp_name)
        st.session_state.temp_name = name if name else "障碍物"
        
        col1, col2 = st.columns(2)
        with col1:
            min_h = st.number_input("最低高度(m)", value=st.session_state.temp_h[0], min_value=0, max_value=200, step=5)
        with col2:
            max_h = st.number_input("最高高度(m)", value=st.session_state.temp_h[1], min_value=min_h+1, max_value=300, step=5)
        st.session_state.temp_h = [min_h, max_h]
        
        # 高度进度条
        st.progress(min(1.0, max_h/300))
        st.caption(f"📊 障碍物高度: {min_h}m - {max_h}m")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 保存", type="primary", use_container_width=True):
                new_obs = Obstacle(st.session_state.temp_obs, min_h, max_h, st.session_state.temp_name)
                st.session_state.obstacles.append(new_obs)
                st.session_state.temp_obs = None
                st.session_state.route_plans = []
                st.session_state.selected_plan = None
                st.rerun()
        with col2:
            if st.button("🗑️ 取消", use_container_width=True):
                st.session_state.temp_obs = None
                st.rerun()
        st.markdown("---")
    
    # ========== A点设置 ==========
    st.markdown("### 🚁 起点 A")
    col1, col2 = st.columns(2)
    with col1:
        la = st.number_input("纬度", value=st.session_state.point_a[0], format="%.6f")
    with col2:
        lo = st.number_input("经度", value=st.session_state.point_a[1], format="%.6f")
    if st.button("📍 设置A点", use_container_width=True):
        st.session_state.point_a = [la, lo]
        st.session_state.route_plans = []
        st.rerun()
    
    # ========== B点设置 ==========
    st.markdown("### 🎯 终点 B")
    col1, col2 = st.columns(2)
    with col1:
        lb = st.number_input("纬度", value=st.session_state.point_b[0], format="%.6f", key="lb")
    with col2:
        lob = st.number_input("经度", value=st.session_state.point_b[1], format="%.6f", key="lob")
    if st.button("🏁 设置B点", use_container_width=True):
        st.session_state.point_b = [lb, lob]
        st.session_state.route_plans = []
        st.rerun()
    
    st.markdown("---")
    
    # ========== 飞行高度 ==========
    st.markdown("### ⚙️ 飞行参数")
    alt = st.slider("飞行高度 (m)", 10, 150, st.session_state.flight_alt)
    st.session_state.flight_alt = alt
    
    # 高度冲突检测
    if st.session_state.obstacles:
        st.markdown("**📊 高度冲突检测**")
        for obs in st.session_state.obstacles:
            if alt < obs.min_h:
                st.success(f"✅ 飞行高度 {alt}m < {obs.name} 底部 {obs.min_h}m，将**绕行**")
            elif alt > obs.max_h:
                st.success(f"✅ 飞行高度 {alt}m > {obs.name} 顶部 {obs.max_h}m，可**飞越**")
            else:
                st.warning(f"⚠️ 飞行高度 {alt}m 与 {obs.name} 高度范围 {obs.min_h}-{obs.max_h}m **冲突**")
    
    st.markdown("---")
    
    # ========== 障碍物列表 ==========
    st.markdown("### 🚧 障碍物列表")
    st.caption(f"共 {len(st.session_state.obstacles)} 个")
    
    for i, obs in enumerate(st.session_state.obstacles):
        if alt < obs.min_h:
            icon = "🔄"
            tip = "将绕行"
        elif alt > obs.max_h:
            icon = "⬆️"
            tip = "可飞越"
        else:
            icon = "⚠️"
            tip = "冲突"
        with st.expander(f"{icon} {obs.name} ({obs.min_h}-{obs.max_h}m) - {tip}"):
            if st.button(f"删除", key=f"del_{i}"):
                st.session_state.obstacles.pop(i)
                st.session_state.route_plans = []
                st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空全部", use_container_width=True):
            st.session_state.obstacles = []
            st.session_state.route_plans = []
            st.rerun()
    
    st.markdown("---")
    
    # ==================== 多航线规划区域 ====================
    st.markdown("## 🗺️ 多航线规划")
    
    # 找碰撞障碍物的函数
    def find_blocking_obstacle(start, end, obstacles, alt):
        """找到阻挡航线的障碍物"""
        for obs in obstacles:
            if alt < obs.min_h:  # 需要绕行
                for t in range(21):
                    t = t/20
                    lat = start[0] + (end[0]-start[0])*t
                    lon = start[1] + (end[1]-start[1])*t
                    if obs.contains([lat, lon]):
                        return obs
        return None
    
    # ========== 生成航线方案按钮 ==========
    if st.button("🎯 生成航线方案", use_container_width=True, type="primary"):
        start = st.session_state.point_a
        end = st.session_state.point_b
        hit_obs = find_blocking_obstacle(start, end, st.session_state.obstacles, alt)
        
        plans = []
        
        if not hit_obs:
            # 无障碍物阻挡
            if alt > max([obs.max_h for obs in st.session_state.obstacles], default=0):
                desc = "✅ 飞行高度高于所有障碍物，直接飞越"
            else:
                desc = "✅ 无障碍物阻挡，直线飞行"
            plans.append({
                'name': '📏 直线路径',
                'points': [start, end],
                'dist': calc_distance(start, end),
                'color': 'blue',
                'desc': desc
            })
        else:
            # 有障碍物，需要绕行
            # 左绕行
            left_pt = hit_obs.get_bypass_left(start, end)
            left_dist = calc_distance(start, left_pt) + calc_distance(left_pt, end)
            plans.append({
                'name': '⬅️ 左绕行',
                'points': [start, left_pt, end],
                'dist': left_dist,
                'color': 'orange',
                'desc': f'从「{hit_obs.name}」左侧绕过'
            })
            
            # 右绕行
            right_pt = hit_obs.get_bypass_right(start, end)
            right_dist = calc_distance(start, right_pt) + calc_distance(right_pt, end)
            plans.append({
                'name': '➡️ 右绕行',
                'points': [start, right_pt, end],
                'dist': right_dist,
                'color': 'purple',
                'desc': f'从「{hit_obs.name}」右侧绕过'
            })
            
            # 最佳航线（取距离短的）
            if left_dist <= right_dist:
                best = {
                    'name': '⭐ 最佳航线',
                    'points': [start, left_pt, end],
                    'dist': left_dist,
                    'color': 'gold',
                    'desc': f'最优路径，比右绕行节省{abs(left_dist-right_dist):.0f}m'
                }
            else:
                best = {
                    'name': '⭐ 最佳航线',
                    'points': [start, right_pt, end],
                    'dist': right_dist,
                    'color': 'gold',
                    'desc': f'最优路径，比左绕行节省{abs(left_dist-right_dist):.0f}m'
                }
            plans.append(best)
        
        st.session_state.route_plans = plans
        if plans:
            st.session_state.selected_plan = plans[0]
        st.rerun()
    
    # ========== 显示三个航线选项 ==========
    if st.session_state.route_plans:
        st.markdown("---")
        st.markdown("### 📋 可选航线方案")
        
        for i, p in enumerate(st.session_state.route_plans):
            # 创建方案卡片
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if "左绕行" in p['name']:
                        st.markdown(f"**🟠 {p['name']}**")
                    elif "右绕行" in p['name']:
                        st.markdown(f"**🟣 {p['name']}**")
                    elif "最佳" in p['name']:
                        st.markdown(f"**⭐ {p['name']}**")
                    else:
                        st.markdown(f"**🔵 {p['name']}**")
                    st.caption(p['desc'])
                with col2:
                    st.metric("距离", f"{p['dist']:.0f}m")
                with col3:
                    st.metric("时间", f"{p['dist']/15:.0f}s")
                
                # 选择按钮
                if st.session_state.selected_plan and st.session_state.selected_plan['name'] == p['name']:
                    st.success("✅ 已选中", icon="✅")
                else:
                    if st.button(f"选择此方案", key=f"select_{i}", use_container_width=True):
                        st.session_state.selected_plan = p
                        st.rerun()
                st.markdown("---")
        
        # 确认使用
        if st.session_state.selected_plan:
            p = st.session_state.selected_plan
            st.info(f"**当前选中: {p['name']}** | 距离: {p['dist']:.0f}m | 时间: {p['dist']/15:.0f}s")
            
            if st.button("✈️ 确认使用此航线", use_container_width=True, type="primary"):
                st.session_state.confirmed_plan = p
                st.success(f"✅ 已确认使用 {p['name']}")
                st.balloons()
    
    # ========== 显示已确认的航线 ==========
    if st.session_state.confirmed_plan:
        st.markdown("---")
        st.markdown("### ✅ 当前航线")
        p = st.session_state.confirmed_plan
        st.success(f"**{p['name']}**")
        st.caption(f"总航程: {p['dist']:.0f}m | 预计时间: {p['dist']/15:.0f}s")

# ==================== 底部提示 ====================
st.markdown("---")
st.caption("💡 **操作步骤**：① 在地图上绘制多边形障碍物 → ② 设置高度和名称 → ③ 点击「生成航线方案」→ ④ 选择左/右/最佳绕行方案 → ⑤ 确认使用")
