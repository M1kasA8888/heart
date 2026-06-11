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
from typing import List, Dict, Optional
from datetime import timedelta
import random

# ==================== 坐标系转换 ====================
class CoordConverter:
    @staticmethod
    def gcj02_to_wgs84(lat, lon):
        return lat, lon  # 简化版，实际需要完整转换
    @staticmethod
    def wgs84_to_gcj02(lat, lon):
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


# ==================== 航线规划 ====================
class FlightPlanner:
    def __init__(self, obstacles, flight_alt):
        self.obstacles = obstacles
        self.flight_alt = flight_alt
    
    def distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        return 6371000 * 2 * math.asin(math.sqrt(
            math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    
    def find_hit_obstacle(self, start, end):
        for obs in self.obstacles:
            if self.flight_alt < obs.min_height:  # 需要绕行
                for i in range(21):
                    t = i/20
                    lat = start[0] + (end[0]-start[0])*t
                    lon = start[1] + (end[1]-start[1])*t
                    if obs.contains_point([lat, lon]):
                        return obs
        return None
    
    def generate_plans(self, start, end):
        plans = []
        hit_obs = self.find_hit_obstacle(start, end)
        
        if not hit_obs:
            plans.append({
                'name': '📏 直线路径',
                'waypoints': [start, end],
                'distance': self.distance(start, end),
                'color': 'blue',
                'desc': '无障碍物'
            })
            return plans
        
        # 左绕行
        left_pt = hit_obs.get_bypass_point(start, end, 'left')
        left_dist = self.distance(start, left_pt) + self.distance(left_pt, end)
        plans.append({
            'name': '⬅️ 左绕行',
            'waypoints': [start, left_pt, end],
            'distance': left_dist,
            'color': 'orange',
            'desc': f'从{hit_obs.name}左侧绕行'
        })
        
        # 右绕行
        right_pt = hit_obs.get_bypass_point(start, end, 'right')
        right_dist = self.distance(start, right_pt) + self.distance(right_pt, end)
        plans.append({
            'name': '➡️ 右绕行',
            'waypoints': [start, right_pt, end],
            'distance': right_dist,
            'color': 'purple',
            'desc': f'从{hit_obs.name}右侧绕行'
        })
        
        # 最佳航线
        if left_dist <= right_dist:
            best = plans[0].copy()
            best['name'] = '⭐ 最佳航线'
            best['color'] = 'gold'
            best['desc'] = f'最短路径，节省{abs(left_dist-right_dist):.0f}m'
        else:
            best = plans[1].copy()
            best['name'] = '⭐ 最佳航线'
            best['color'] = 'gold'
            best['desc'] = f'最短路径，节省{abs(left_dist-right_dist):.0f}m'
        plans.append(best)
        
        return plans


# ==================== 持久化 ====================
class ObstaclePersistence:
    CONFIG_FILE = "obstacle_config.json"
    @classmethod
    def save(cls, obstacles):
        data = [{'points': o.points, 'min_height': o.min_height, 
                 'max_height': o.max_height, 'name': o.name} for o in obstacles]
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(data, f)
    @classmethod
    def load(cls):
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r') as f:
                data = json.load(f)
            return [Obstacle3D(d['points'], d['min_height'], d['max_height'], d['name']) for d in data]
        return []


# ==================== 页面配置 ====================
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
if 'confirmed_plan' not in st.session_state:
    st.session_state.confirmed_plan = None

st.title("🛰️ 无人机智能航线规划系统")
st.markdown("南京科技职业学院 | 多航线选择（左绕行/右绕行/最佳航线）")
st.markdown("---")

# 布局
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🗺️ 地图")
    
    m = folium.Map(location=CAMPUS, zoom_start=17, control_scale=True)
    folium.TileLayer(
        tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
        attr='高德卫星', subdomains=['1','2','3','4']
    ).add_to(m)
    
    # 学院标记
    folium.Marker(CAMPUS, popup="🏫 南京科技职业学院", 
                 icon=folium.Icon(color='red', icon='university')).add_to(m)
    
    # 绘制障碍物
    for obs in st.session_state.obstacles:
        if len(obs.points) >= 3:
            color = 'red' if st.session_state.flight_alt < obs.min_height else 'green'
            folium.Polygon(locations=obs.points, color=color, weight=2,
                          fill=True, fill_color=color, fill_opacity=0.3,
                          popup=f"{obs.name}\n{obs.min_height}-{obs.max_height}m").add_to(m)
    
    # 绘制A/B点
    folium.Marker(st.session_state.point_a, popup="🚁 起点A", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(st.session_state.point_b, popup="🎯 终点B", icon=folium.Icon(color='red')).add_to(m)
    
    # 绘制选中的航线
    if st.session_state.selected_plan:
        plan = st.session_state.selected_plan
        wps = plan['waypoints']
        folium.PolyLine(wps, color=plan['color'], weight=4, opacity=0.9,
                       popup=f"{plan['name']}\n{plan['distance']:.0f}m").add_to(m)
        if len(wps) > 2:
            folium.Marker(wps[1], popup="绕行点", icon=folium.Icon(color='purple')).add_to(m))
    
    # 绘图工具
    plugins.Draw(draw_options={'polygon': {'allowIntersection': False}}).add_to(m)
    plugins.MeasureControl().add_to(m)
    
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
    
    # === 新建障碍物 ===
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
            if st.button("✅ 保存", type="primary"):
                obs = Obstacle3D(st.session_state.temp_obs, min_h, max_h, st.session_state.temp_name)
                st.session_state.obstacles.append(obs)
                st.session_state.temp_obs = None
                st.session_state.route_plans = []
                st.rerun()
        with col2:
            if st.button("🗑️ 取消"):
                st.session_state.temp_obs = None
                st.rerun()
        st.markdown("---")
    
    # === 起点 ===
    st.markdown("### 🚁 起点 A")
    col1, col2 = st.columns(2)
    with col1:
        lat_a = st.number_input("纬度", value=st.session_state.point_a[0], format="%.6f")
    with col2:
        lon_a = st.number_input("经度", value=st.session_state.point_a[1], format="%.6f")
    if st.button("设置A点", use_container_width=True):
        st.session_state.point_a = [lat_a, lon_a]
        st.rerun()
    
    # === 终点 ===
    st.markdown("### 🎯 终点 B")
    col1, col2 = st.columns(2)
    with col1:
        lat_b = st.number_input("纬度", value=st.session_state.point_b[0], format="%.6f", key="lat_b")
    with col2:
        lon_b = st.number_input("经度", value=st.session_state.point_b[1], format="%.6f", key="lon_b")
    if st.button("设置B点", use_container_width=True):
        st.session_state.point_b = [lat_b, lon_b]
        st.rerun()
    
    st.markdown("---")
    
    # === 飞行高度 ===
    st.markdown("### ⚙️ 飞行参数")
    flight_alt = st.slider("飞行高度 (m)", 10, 150, st.session_state.flight_alt)
    st.session_state.flight_alt = flight_alt
    
    st.info("💡 高度 < 障碍物 → 绕行 | 高度 > 障碍物 → 飞越")
    
    st.markdown("---")
    
    # === 障碍物列表 ===
    st.markdown("### 🚧 障碍物列表")
    for i, obs in enumerate(st.session_state.obstacles):
        if flight_alt < obs.min_height:
            icon = "🔄"
        elif flight_alt > obs.max_height:
            icon = "⬆️"
        else:
            icon = "⚠️"
        with st.expander(f"{icon} {obs.name} ({obs.min_height}-{obs.max_height}m)"):
            if st.button(f"删除", key=f"del_{i}"):
                st.session_state.obstacles.pop(i)
                st.session_state.route_plans = []
                st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存配置"):
            ObstaclePersistence.save(st.session_state.obstacles)
            st.success("已保存")
    with col2:
        if st.button("📂 加载配置"):
            st.session_state.obstacles = ObstaclePersistence.load()
            st.rerun()
    
    st.markdown("---")
    st.markdown("## 🗺️ 多航线规划")
    
    # ========== 关键：生成航线方案按钮 ==========
    if st.button("🎯 生成航线方案", use_container_width=True, type="primary"):
        with st.spinner("正在规划航线..."):
            planner = FlightPlanner(st.session_state.obstacles, flight_alt)
            plans = planner.generate_plans(st.session_state.point_a, st.session_state.point_b)
            st.session_state.route_plans = plans
            if plans:
                st.session_state.selected_plan = plans[0]
                st.success(f"✅ 已生成 {len(plans)} 条航线方案")
            else:
                st.error("无法生成航线")
            st.rerun()
    
    # ========== 显示三个航线选项 ==========
    if st.session_state.route_plans:
        st.markdown("---")
        st.markdown("### 📋 可选方案（点击选择）")
        
        for i, plan in enumerate(st.session_state.route_plans):
            # 创建卡片
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    # 根据不同类型显示不同图标
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
                
                # 高亮当前选中的方案
                if st.session_state.selected_plan and st.session_state.selected_plan['name'] == plan['name']:
                    st.button("✅ 已选中", key=f"selected_{i}", disabled=True, use_container_width=True)
                else:
                    if st.button(f"选择此方案", key=f"select_{i}", use_container_width=True):
                        st.session_state.selected_plan = plan
                        st.success(f"已选择: {plan['name']}")
                        st.rerun()
                st.markdown("---")
        
        # 确认使用航线
        if st.session_state.selected_plan:
            st.markdown("### ✅ 当前选中")
            plan = st.session_state.selected_plan
            st.info(f"**{plan['name']}** | 距离: {plan['distance']:.0f}m | 时间: {plan['distance']/15:.0f}s")
            
            if st.button("✈️ 确认使用此航线", use_container_width=True, type="primary"):
                st.session_state.confirmed_plan = plan
                st.success(f"✅ 已确认使用 {plan['name']}，可前往飞行监控页面")
                
                # 保存到flight_plan供飞行监控使用
                st.session_state.flight_plan = {
                    'waypoints': plan['waypoints'],
                    'total_distance': plan['distance'],
                    'estimated_time': plan['distance']/15,
                    'path_type': plan['name']
                }
    
    # 显示已确认的航线
    if st.session_state.confirmed_plan:
        st.markdown("---")
        st.success(f"✈️ 当前使用航线: **{st.session_state.confirmed_plan['name']}**")


# ==================== 飞行监控页面（简化） ====================
st.markdown("---")
st.subheader("📡 飞行监控")

if st.session_state.get('flight_plan'):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ 开始飞行", type="primary"):
            st.info("飞行模拟功能开发中...")
else:
    st.info("请先在左侧「航线规划」页面确认航线")
