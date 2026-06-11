import streamlit as st
import folium
from streamlit_folium import st_folium
from folium import plugins
import pandas as pd
import numpy as np
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
import math
import json
import os
import random

# ==================== 坐标系转换 ====================
class CoordConverter:
    @staticmethod
    def gcj02_to_wgs84(lat, lon):
        return lat, lon


# ==================== 3D障碍物 ====================
class Obstacle3D:
    def __init__(self, points, min_h=0, max_h=100, name="障碍物"):
        self.points = points
        self.min_height = min_h
        self.max_height = max_h
        self.name = name
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.center_lat = (min(lats) + max(lats)) / 2
        self.center_lon = (min(lons) + max(lons)) / 2
    
    def contains_point(self, point):
        x, y = point[1], point[0]
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i][1], self.points[i][0]
            x2, y2 = self.points[(i+1)%n][1], self.points[(i+1)%n][0]
            if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1)+x1):
                inside = not inside
        return inside
    
    def get_bypass_point(self, start, end, side):
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        perp_x = -dy
        perp_y = dx
        offset = 0.003
        if side == 'left':
            return [self.center_lat + perp_y * offset, self.center_lon + perp_x * offset]
        else:
            return [self.center_lat - perp_y * offset, self.center_lon - perp_x * offset]


# ==================== 持久化 ====================
class ObstaclePersistence:
    CONFIG_FILE = "obstacle_config.json"
    @classmethod
    def save(cls, obstacles):
        data = [{'points': o.points, 'min_height': o.min_height, 'max_height': o.max_height, 'name': o.name} for o in obstacles]
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(data, f)
    @classmethod
    def load(cls):
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r') as f:
                data = json.load(f)
            return [Obstacle3D(d['points'], d['min_height'], d['max_height'], d['name']) for d in data]
        return []


# ==================== 主程序 ====================
st.set_page_config(page_title="无人机航线规划", page_icon="🛰️", layout="wide")

CAMPUS = [32.234097, 118.749413]

# 初始化
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = ObstaclePersistence.load()
if 'temp_obs' not in st.session_state:
    st.session_state.temp_obs = None
if 'temp_height' not in st.session_state:
    st.session_state.temp_height = [0, 30]
if 'temp_name' not in st.session_state:
    st.session_state.temp_name = "建筑物"
if 'point_a' not in st.session_state:
    st.session_state.point_a = [32.2323, 118.749]
if 'point_b' not in st.session_state:
    st.session_state.point_b = [32.2344, 118.749]
if 'flight_alt' not in st.session_state:
    st.session_state.flight_alt = 20
if 'route_plans' not in st.session_state:
    st.session_state.route_plans = []
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None
if 'show_plans' not in st.session_state:
    st.session_state.show_plans = False

st.title("🛰️ 无人机智能航线规划系统")
st.markdown("**南京科技职业学院** | 多航线选择（左绕行/右绕行/最佳航线）")
st.markdown("---")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🗺️ 卫星地图")
    
    m = folium.Map(location=CAMPUS, zoom_start=17, control_scale=True)
    folium.TileLayer(
        tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
        attr='高德卫星', subdomains=['1','2','3','4'], name='卫星'
    ).add_to(m)
    
    folium.Marker(CAMPUS, popup="🏫 南京科技职业学院", icon=folium.Icon(color='red', icon='university')).add_to(m)
    
    # 障碍物
    for obs in st.session_state.obstacles:
        if len(obs.points) >= 3:
            color = 'red' if st.session_state.flight_alt < obs.min_height else 'green'
            folium.Polygon(locations=obs.points, color=color, weight=2,
                          fill=True, fill_color=color, fill_opacity=0.3,
                          popup=f"{obs.name}\n{obs.min_height}-{obs.max_height}m").add_to(m)
    
    # A/B点
    folium.Marker(st.session_state.point_a, popup="🚁 起点A", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(st.session_state.point_b, popup="🎯 终点B", icon=folium.Icon(color='red')).add_to(m)
    
    # 选中的航线
    if st.session_state.selected_plan:
        plan = st.session_state.selected_plan
        wps = plan['waypoints']
        folium.PolyLine(wps, color=plan['color'], weight=4, opacity=0.9,
                       popup=f"{plan['name']}\n{plan['distance']:.0f}m").add_to(m)
        if len(wps) > 2:
            folium.Marker(wps[1], popup="绕行点", icon=folium.Icon(color='purple')).add_to(m)
    
    plugins.Draw(draw_options={'polygon': {'allowIntersection': False}}).add_to(m)
    plugins.MeasureControl().add_to(m)
    folium.LayerControl().add_to(m)
    
    output = st_folium(m, width=700, height=500, key="map")
    
    if output and output.get('last_active_drawing'):
        draw = output['last_active_drawing']
        if draw and draw['geometry']['type'] == 'Polygon':
            coords = draw['geometry']['coordinates'][0]
            points = [[c[1], c[0]] for c in coords]
            if len(points) >= 3:
                st.session_state.temp_obs = points
                st.success(f"✅ 已绘制 {len(points)} 个点")

with col_right:
    st.subheader("🎯 控制面板")
    
    # ========== 新建障碍物 ==========
    if st.session_state.temp_obs:
        st.markdown("### 🆕 新建障碍物")
        name = st.text_input("名称", value=st.session_state.temp_name)
        st.session_state.temp_name = name if name else "建筑物"
        
        col1, col2 = st.columns(2)
        with col1:
            min_h = st.number_input("最低(m)", value=st.session_state.temp_height[0], min_value=0, max_value=200)
        with col2:
            max_h = st.number_input("最高(m)", value=st.session_state.temp_height[1], min_value=min_h+1, max_value=300)
        st.session_state.temp_height = [min_h, max_h]
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 保存", type="primary", use_container_width=True):
                obs = Obstacle3D(st.session_state.temp_obs, min_h, max_h, st.session_state.temp_name)
                st.session_state.obstacles.append(obs)
                st.session_state.temp_obs = None
                st.rerun()
        with col2:
            if st.button("🗑️ 取消", use_container_width=True):
                st.session_state.temp_obs = None
                st.rerun()
        st.markdown("---")
    
    # ========== A点 ==========
    st.markdown("### 🚁 起点 A")
    col1, col2 = st.columns(2)
    with col1:
        lat_a = st.number_input("纬度", value=st.session_state.point_a[0], format="%.6f")
    with col2:
        lon_a = st.number_input("经度", value=st.session_state.point_a[1], format="%.6f")
    if st.button("📍 设置A点", use_container_width=True):
        st.session_state.point_a = [lat_a, lon_a]
        st.rerun()
    
    # ========== B点 ==========
    st.markdown("### 🎯 终点 B")
    col1, col2 = st.columns(2)
    with col1:
        lat_b = st.number_input("纬度", value=st.session_state.point_b[0], format="%.6f", key="lat_b")
    with col2:
        lon_b = st.number_input("经度", value=st.session_state.point_b[1], format="%.6f", key="lon_b")
    if st.button("🏁 设置B点", use_container_width=True):
        st.session_state.point_b = [lat_b, lon_b]
        st.rerun()
    
    st.markdown("---")
    
    # ========== 飞行高度 ==========
    st.markdown("### ⚙️ 飞行参数")
    flight_alt = st.slider("飞行高度 (m)", 10, 150, st.session_state.flight_alt)
    st.session_state.flight_alt = flight_alt
    
    st.markdown("---")
    
    # ========== 障碍物管理 ==========
    st.markdown("### 🚧 障碍物列表")
    if st.session_state.obstacles:
        for i, obs in enumerate(st.session_state.obstacles):
            with st.expander(f"🚧 {obs.name} ({obs.min_height}-{obs.max_height}m)"):
                if st.button(f"删除", key=f"del_{i}"):
                    st.session_state.obstacles.pop(i)
                    st.rerun()
    else:
        st.caption("暂无障碍物，请在地图上绘制多边形")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存配置", use_container_width=True):
            ObstaclePersistence.save(st.session_state.obstacles)
            st.success("已保存")
    with col2:
        if st.button("📂 加载配置", use_container_width=True):
            st.session_state.obstacles = ObstaclePersistence.load()
            st.rerun()
    
    st.markdown("---")
    
    # ============================================================
    # ========== 重点：多航线规划区域 ==========
    # ============================================================
    st.markdown("## 🗺️ 多航线规划")
    
    # 距离计算函数
    def calc_distance(p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        return 6371000 * 2 * math.asin(math.sqrt(
            math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    
    # 找碰撞障碍物
    def find_collision(start, end, obstacles, alt):
        for obs in obstacles:
            if alt < obs.min_height:
                for i in range(21):
                    t = i/20
                    lat = start[0] + (end[0]-start[0])*t
                    lon = start[1] + (end[1]-start[1])*t
                    if obs.contains_point([lat, lon]):
                        return obs
        return None
    
    # 生成航线方案按钮
    if st.button("🎯 生成航线方案", use_container_width=True, type="primary"):
        start = st.session_state.point_a
        end = st.session_state.point_b
        obstacles = st.session_state.obstacles
        alt = st.session_state.flight_alt
        
        # 找碰撞的障碍物
        hit_obs = find_collision(start, end, obstacles, alt)
        
        plans = []
        
        if not hit_obs:
            # 无碰撞，直线
            dist = calc_distance(start, end)
            plans.append({
                'name': '📏 直线路径',
                'waypoints': [start, end],
                'distance': dist,
                'color': 'blue',
                'desc': '✅ 无障碍物，直接飞行'
            })
        else:
            # 左绕行
            left_pt = hit_obs.get_bypass_point(start, end, 'left')
            left_dist = calc_distance(start, left_pt) + calc_distance(left_pt, end)
            plans.append({
                'name': '⬅️ 左绕行',
                'waypoints': [start, left_pt, end],
                'distance': left_dist,
                'color': 'orange',
                'desc': f'从「{hit_obs.name}」左侧绕过'
            })
            
            # 右绕行
            right_pt = hit_obs.get_bypass_point(start, end, 'right')
            right_dist = calc_distance(start, right_pt) + calc_distance(right_pt, end)
            plans.append({
                'name': '➡️ 右绕行',
                'waypoints': [start, right_pt, end],
                'distance': right_dist,
                'color': 'purple',
                'desc': f'从「{hit_obs.name}」右侧绕过'
            })
            
            # 最佳航线
            best = plans[0].copy() if left_dist <= right_dist else plans[1].copy()
            best['name'] = '⭐ 最佳航线'
            best['color'] = 'gold'
            best['desc'] = f'最优路径，节省{abs(left_dist-right_dist):.0f}m'
            plans.append(best)
        
        st.session_state.route_plans = plans
        st.session_state.show_plans = True
        if plans:
            st.session_state.selected_plan = plans[0]
        st.rerun()
    
    # ========== 显示三个航线选项 ==========
    if st.session_state.show_plans and st.session_state.route_plans:
        st.markdown("---")
        st.markdown("### 📋 可选方案")
        
        for i, plan in enumerate(st.session_state.route_plans):
            # 创建卡片样式
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if "左绕行" in plan['name']:
                        st.markdown(f"**🟠 {plan['name']}**")
                    elif "右绕行" in plan['name']:
                        st.markdown(f"**🟣 {plan['name']}**")
                    elif "最佳" in plan['name']:
                        st.markdown(f"**⭐ {plan['name']}**")
                    else:
                        st.markdown(f"**🔵 {plan['name']}**")
                    st.caption(plan['desc'])
                with col2:
                    st.metric("距离", f"{plan['distance']:.0f}m")
                with col3:
                    st.metric("时间", f"{plan['distance']/15:.0f}s")
                
                if st.session_state.selected_plan and st.session_state.selected_plan['name'] == plan['name']:
                    st.button("✅ 已选中", key=f"sel_{i}", disabled=True, use_container_width=True)
                else:
                    if st.button(f"选择此方案", key=f"select_{i}", use_container_width=True):
                        st.session_state.selected_plan = plan
                        st.success(f"已选择: {plan['name']}")
                        st.rerun()
                st.markdown("---")
        
        # 确认使用
        if st.session_state.selected_plan:
            plan = st.session_state.selected_plan
            st.info(f"**当前选中: {plan['name']}** | 距离: {plan['distance']:.0f}m | 时间: {plan['distance']/15:.0f}s")
            
            if st.button("✈️ 确认使用此航线", use_container_width=True, type="primary"):
                st.session_state.flight_plan = {
                    'waypoints': plan['waypoints'],
                    'total_distance': plan['distance'],
                    'estimated_time': plan['distance']/15,
                    'path_type': plan['name']
                }
                st.success(f"✅ 已确认使用 {plan['name']}")
                st.balloons()
    
    # 显示已确认的航线
    if st.session_state.get('flight_plan'):
        st.markdown("---")
        st.success(f"✈️ **当前航线: {st.session_state.flight_plan['path_type']}**")
        st.caption(f"总航程: {st.session_state.flight_plan['total_distance']:.0f}m | 预计时间: {st.session_state.flight_plan['estimated_time']:.0f}s")
