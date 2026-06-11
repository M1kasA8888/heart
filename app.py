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

# ==================== 坐标系转换模块 ====================
class CoordConverter:
    a = 6378245.0
    ee = 0.00669342162296594323
    
    @staticmethod
    def _transform_lat(lon, lat):
        ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
        ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 *
                math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 *
                math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 *
                math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    @staticmethod
    def _transform_lon(lon, lat):
        ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + \
              0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
        ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 *
                math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lon * math.pi) + 40.0 *
                math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 *
                math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    
    @staticmethod
    def _out_of_china(lon, lat):
        if lon < 72.004 or lon > 137.8347:
            return True
        if lat < 0.8293 or lat > 55.8271:
            return True
        return False
    
    @classmethod
    def gcj02_to_wgs84(cls, lat, lon):
        if cls._out_of_china(lon, lat):
            return lat, lon
        dlat = cls._transform_lat(lon - 105.0, lat - 35.0)
        dlon = cls._transform_lon(lon - 105.0, lat - 35.0)
        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - cls.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((cls.a * (1 - cls.ee)) / (magic * sqrtmagic) * math.pi)
        dlon = (dlon * 180.0) / (cls.a / sqrtmagic * math.cos(radlat) * math.pi)
        mg_lat = lat + dlat
        mg_lon = lon + dlon
        return mg_lat, mg_lon


# ==================== 3D障碍物类 ====================
class Obstacle3D:
    def __init__(self, points: List[List[float]], min_height: float = 0, 
                 max_height: float = 100, name: str = "障碍物"):
        self.points = points
        self.min_height = min_height
        self.max_height = max_height
        self.name = name
        self.created_time = datetime.now()
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.min_lat = min(lats)
        self.max_lat = max(lats)
        self.min_lon = min(lons)
        self.max_lon = max(lons)
        self.center_lat = (self.min_lat + self.max_lat) / 2
        self.center_lon = (self.min_lon + self.max_lon) / 2
    
    def to_dict(self) -> dict:
        return {
            'points': self.points,
            'min_height': self.min_height,
            'max_height': self.max_height,
            'name': self.name,
            'created_time': self.created_time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        obstacle = cls(
            data['points'],
            data.get('min_height', 0),
            data.get('max_height', 100),
            data.get('name', '障碍物')
        )
        return obstacle
    
    def contains_point(self, point: List[float]) -> bool:
        x, y = point[1], point[0]
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i][1], self.points[i][0]
            x2, y2 = self.points[(i + 1) % n][1], self.points[(i + 1) % n][0]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def get_bypass_point(self, start: List[float], end: List[float], side: str) -> List[float]:
        """获取绕行点
        side: 'left', 'right', 'best'
        """
        # 计算从起点到终点的方向
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        # 垂直方向
        perp_x = -dy
        perp_y = dx
        
        # 偏移量（0.003度约330米）
        offset = 0.003
        
        if side == 'left':
            # 左侧绕行
            lat = self.center_lat + perp_y * offset
            lon = self.center_lon + perp_x * offset
        elif side == 'right':
            # 右侧绕行
            lat = self.center_lat - perp_y * offset
            lon = self.center_lon - perp_x * offset
        else:
            # 最佳航线：取距离短的一侧
            left_point = [self.center_lat + perp_y * offset, self.center_lon + perp_x * offset]
            right_point = [self.center_lat - perp_y * offset, self.center_lon - perp_x * offset]
            dist_left = self._distance(start, left_point) + self._distance(left_point, end)
            dist_right = self._distance(start, right_point) + self._distance(right_point, end)
            if dist_left <= dist_right:
                return left_point
            else:
                return right_point
        
        return [lat, lon]
    
    def _distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def get_description(self) -> str:
        return f"🚧 {self.name}\n高度: {self.min_height}-{self.max_height}m"


# ==================== 障碍物持久化管理 ====================
class ObstaclePersistence:
    CONFIG_FILE = "obstacle_config.json"
    
    @classmethod
    def save_obstacles(cls, obstacles: List[Obstacle3D]):
        config = {
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'obstacles': [obs.to_dict() for obs in obstacles],
            'count': len(obstacles)
        }
        try:
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True, config
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def load_obstacles(cls):
        if not os.path.exists(cls.CONFIG_FILE):
            return [], None
        try:
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            obstacles = [Obstacle3D.from_dict(obs_data) for obs_data in config.get('obstacles', [])]
            return obstacles, config
        except:
            return [], None
    
    @classmethod
    def get_config_status(cls):
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return {'exists': True, 'count': config.get('count', 0), 'save_time': config.get('save_time', '未知')}
            except:
                return {'exists': False}
        return {'exists': False}


# ==================== 航线方案类 ====================
class RoutePlan:
    def __init__(self, name: str, waypoints: List[List[float]], total_distance: float, 
                 color: str, description: str, side: str = ""):
        self.name = name
        self.waypoints = waypoints
        self.total_distance = total_distance
        self.estimated_time = total_distance / 15
        self.color = color
        self.description = description
        self.side = side


# ==================== 智能航线规划 ====================
class FlightPlanner:
    def __init__(self, obstacles: List[Obstacle3D], flight_altitude: float):
        self.obstacles = obstacles
        self.flight_altitude = flight_altitude
    
    def calculate_distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def line_hits_obstacle(self, start, end):
        """检查直线是否碰到障碍物"""
        for obs in self.obstacles:
            # 如果飞行高度低于障碍物底部，需要检查2D投影
            if self.flight_altitude < obs.min_height:
                num_samples = 20
                for i in range(num_samples + 1):
                    t = i / num_samples
                    lat = start[0] + (end[0] - start[0]) * t
                    lon = start[1] + (end[1] - start[1]) * t
                    if obs.contains_point([lat, lon]):
                        return True, obs
            # 如果飞行高度高于障碍物顶部，可以直接飞越
            elif self.flight_altitude > obs.max_height:
                continue
            # 如果飞行高度在障碍物高度范围内，需要检查
            else:
                num_samples = 20
                for i in range(num_samples + 1):
                    t = i / num_samples
                    lat = start[0] + (end[0] - start[0]) * t
                    lon = start[1] + (end[1] - start[1]) * t
                    if obs.contains_point([lat, lon]):
                        return True, obs
        return False, None
    
    def generate_route_plans(self, start: List[float], end: List[float]) -> List[RoutePlan]:
        """生成三个航线方案"""
        plans = []
        
        # 检查直线是否有障碍物
        hit, hit_obs = self.line_hits_obstacle(start, end)
        
        if not hit:
            # 无障碍物，返回直线
            distance = self.calculate_distance(start, end)
            plans.append(RoutePlan(
                name="📏 直线路径",
                waypoints=[start, end],
                total_distance=distance,
                color="blue",
                description="✅ 无障碍物，直接飞行",
                side="straight"
            ))
            return plans
        
        # 有障碍物，生成三个绕行方案
        if hit_obs:
            # 左绕行
            left_point = hit_obs.get_bypass_point(start, end, "left")
            left_waypoints = [start, left_point, end]
            left_distance = (self.calculate_distance(start, left_point) + 
                           self.calculate_distance(left_point, end))
            plans.append(RoutePlan(
                name="⬅️ 左绕行",
                waypoints=left_waypoints,
                total_distance=left_distance,
                color="orange",
                description=f"从「{hit_obs.name}」左侧绕过",
                side="left"
            ))
            
            # 右绕行
            right_point = hit_obs.get_bypass_point(start, end, "right")
            right_waypoints = [start, right_point, end]
            right_distance = (self.calculate_distance(start, right_point) + 
                            self.calculate_distance(right_point, end))
            plans.append(RoutePlan(
                name="➡️ 右绕行",
                waypoints=right_waypoints,
                total_distance=right_distance,
                color="purple",
                description=f"从「{hit_obs.name}」右侧绕过",
                side="right"
            ))
            
            # 最佳航线（取距离短的）
            if left_distance <= right_distance:
                best_waypoints = left_waypoints
                best_distance = left_distance
                best_side = "left"
            else:
                best_waypoints = right_waypoints
                best_distance = right_distance
                best_side = "right"
            
            plans.append(RoutePlan(
                name="⭐ 最佳航线",
                waypoints=best_waypoints,
                total_distance=best_distance,
                color="gold",
                description=f"最优路径，节省{abs(left_distance - right_distance):.0f}m",
                side=best_side
            ))
        
        # 按距离排序
        plans.sort(key=lambda x: x.total_distance)
        return plans


# ==================== 无人机模拟器 ====================
class DroneSimulator:
    def __init__(self, waypoints: List[List[float]], speed: float = 15, altitude: float = 50):
        self.waypoints = waypoints
        self.speed = speed
        self.altitude = altitude
        self.current_index = 0
        self.position = waypoints[0].copy()
        self.completed = 0.0
        self.total = self._calc_total()
        self.flying = True
    
    def _distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def _calc_total(self):
        total = 0
        for i in range(len(self.waypoints) - 1):
            total += self._distance(self.waypoints[i], self.waypoints[i+1])
        return total
    
    def update(self, dt=0.1):
        if not self.flying or self.current_index >= len(self.waypoints) - 1:
            return False
        target = self.waypoints[self.current_index + 1]
        dist = self._distance(self.position, target)
        step = self.speed * dt
        if dist <= step:
            self.position = target
            self.completed += dist
            self.current_index += 1
        else:
            ratio = step / dist
            new_lat = self.position[0] + (target[0] - self.position[0]) * ratio
            new_lon = self.position[1] + (target[1] - self.position[1]) * ratio
            self.position = [new_lat, new_lon]
            self.completed += step
        return self.current_index < len(self.waypoints) - 1
    
    def get_status(self):
        progress = (self.completed / self.total * 100) if self.total > 0 else 0
        return {
            'position': self.position,
            'current': self.current_index + 1,
            'total': len(self.waypoints),
            'remaining': self.total - self.completed,
            'progress': progress,
            'altitude': self.altitude
        }


# ==================== 心跳监控 ====================
class HeartbeatMonitor:
    def __init__(self):
        self.seq = 0
        self.send_log = []
        self.recv_log = []
        self.last_time = None
    
    def send(self):
        self.seq += 1
        send_time = datetime.now()
        delay = random.uniform(5, 50)
        recv_time = send_time + timedelta(milliseconds=delay)
        self.send_log.append({'seq': self.seq, 'time': send_time})
        self.recv_log.append({'seq': self.seq, 'send': send_time, 'recv': recv_time, 'delay': round(delay, 2)})
        self.last_time = recv_time
        return {'seq': self.seq, 'delay': round(delay, 2)}
    
    def get_status(self):
        connected = self.last_time and (datetime.now() - self.last_time).total_seconds() < 3
        return {
            'sent': len(self.send_log),
            'recv': len(self.recv_log),
            'connected': connected,
            'rate': 60 if connected else 0
        }
    
    def get_recent(self, n=8):
        return [{'seq': h['seq'], 'send': h['send'].strftime("%H:%M:%S.%f")[:-3],
                 'recv': h['recv'].strftime("%H:%M:%S.%f")[:-3], 'delay': h['delay']} 
                for h in self.recv_log[-n:]]


# ==================== 页面配置 ====================
st.set_page_config(page_title="无人机智能监控系统 - 南京科技职业学院", page_icon="🛰️", layout="wide")

CAMPUS = [32.234097, 118.749413]

# 初始化
if 'page' not in st.session_state:
    st.session_state.page = "航线规划"
if 'obstacles' not in st.session_state:
    loaded, _ = ObstaclePersistence.load_obstacles()
    st.session_state.obstacles = loaded if loaded else []
if 'temp_obs' not in st.session_state:
    st.session_state.temp_obs = None
if 'temp_name' not in st.session_state:
    st.session_state.temp_name = "建筑物"
if 'temp_height' not in st.session_state:
    st.session_state.temp_height = [0, 30]
if 'route_plans' not in st.session_state:
    st.session_state.route_plans = []
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None
if 'point_a' not in st.session_state:
    st.session_state.point_a = [32.2323, 118.749]
if 'point_b' not in st.session_state:
    st.session_state.point_b = [32.2344, 118.749]
if 'flight_alt' not in st.session_state:
    st.session_state.flight_alt = 20
if 'flight_plan' not in st.session_state:
    st.session_state.flight_plan = None
if 'hb' not in st.session_state:
    st.session_state.hb = HeartbeatMonitor()
if 'flying' not in st.session_state:
    st.session_state.flying = False
if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'start_t' not in st.session_state:
    st.session_state.start_t = None
if 'alt_data' not in st.session_state:
    st.session_state.alt_data = []


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🛰️ 无人机系统")
    st.caption("南京科技职业学院")
    st.markdown("---")
    
    page = st.radio("📱 功能页面", ["🗺️ 航线规划", "📡 飞行监控"], label_visibility="collapsed")
    st.session_state.page = page
    
    st.markdown("---")
    st.subheader("🌐 坐标系")
    coord = st.selectbox("输入坐标系", ["WGS-84", "GCJ-02 (高德/百度)"])
    st.session_state.coord_type = "WGS-84" if coord == "WGS-84" else "GCJ-02"
    
    st.markdown("---")
    st.subheader("📊 系统状态")
    st.success("✅ 系统正常")
    status = ObstaclePersistence.get_config_status()
    if status['exists']:
        st.info(f"💾 {status['count']}个障碍物 | {status['save_time']}")


# ==================== 航线规划页面 ====================
if st.session_state.page == "🗺️ 航线规划":
    st.title("🛰️ 航线规划")
    st.markdown("设置起降点、3D障碍物高度，**多航线选择**（左绕行/右绕行/最佳航线）")
    st.markdown("---")
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.subheader("🗺️ 卫星地图")
        
        m = folium.Map(location=CAMPUS, zoom_start=17, control_scale=True)
        folium.TileLayer(
            tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
            attr='高德卫星', subdomains=['1','2','3','4'], name='卫星'
        ).add_to(m)
        
        # 学院标记
        folium.Marker(CAMPUS, popup="🏫 南京科技职业学院", 
                     icon=folium.Icon(color='red', icon='university')).add_to(m)
        
        # 绘制障碍物
        for obs in st.session_state.obstacles:
            if len(obs.points) >= 3:
                pts = obs.points
                if st.session_state.coord_type == "GCJ-02":
                    pts = [CoordConverter.gcj02_to_wgs84(p[0], p[1]) for p in obs.points]
                color = 'red' if st.session_state.flight_alt < obs.min_height else 'green'
                folium.Polygon(locations=[[p[0],p[1]] for p in pts], color=color, weight=2,
                              fill=True, fill_color=color, fill_opacity=0.3,
                              popup=f"{obs.name}\n{obs.min_height}-{obs.max_height}m").add_to(m)
        
        # A/B点
        a_pt = st.session_state.point_a
        b_pt = st.session_state.point_b
        if st.session_state.coord_type == "GCJ-02":
            a_pt = CoordConverter.gcj02_to_wgs84(a_pt[0], a_pt[1])
            b_pt = CoordConverter.gcj02_to_wgs84(b_pt[0], b_pt[1])
        folium.Marker([a_pt[0], a_pt[1]], popup="🚁 起点A", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([b_pt[0], b_pt[1]], popup="🎯 终点B", icon=folium.Icon(color='red')).add_to(m)
        
        # 绘制所有航线方案
        if st.session_state.route_plans:
            for i, plan in enumerate(st.session_state.route_plans):
                wps = plan.waypoints
                if st.session_state.coord_type == "GCJ-02":
                    wps = [CoordConverter.gcj02_to_wgs84(wp[0], wp[1]) for wp in wps]
                width = 4 if st.session_state.selected_plan and st.session_state.selected_plan.name == plan.name else 2
                opacity = 0.9 if st.session_state.selected_plan and st.session_state.selected_plan.name == plan.name else 0.4
                folium.PolyLine([[wp[0],wp[1]] for wp in wps], color=plan.color, weight=width, opacity=opacity,
                               popup=f"{plan.name}\n{plan.total_distance:.0f}m").add_to(m)
                if len(wps) > 2:
                    folium.Marker([wps[1][0], wps[1][1]], popup="绕行点", 
                                 icon=folium.Icon(color='purple', icon='refresh')).add_to(m)
        
        # 绘图工具
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
                    st.success(f"✅ 已绘制 {len(points)} 个点，右侧设置高度和名称")
    
    with col_right:
        st.subheader("🎯 控制面板")
        
        # 新建障碍物
        if st.session_state.temp_obs:
            st.markdown("### 🆕 新建障碍物")
            name = st.text_input("名称", value=st.session_state.temp_name)
            st.session_state.temp_name = name if name else "建筑物"
            col1, col2 = st.columns(2)
            with col1:
                min_h = st.number_input("最低高度(m)", value=st.session_state.temp_height[0], min_value=0, max_value=200)
            with col2:
                max_h = st.number_input("最高高度(m)", value=st.session_state.temp_height[1], min_value=min_h+1, max_value=300)
            st.session_state.temp_height = [min_h, max_h]
            st.progress(min(1.0, max_h/300))
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
        
        # 定位
        if st.button("📍 定位学院"):
            st.success("已定位")
            st.rerun()
        st.markdown("---")
        
        # A点设置
        st.markdown("### 🚁 起点A")
        col1, col2 = st.columns(2)
        with col1:
            lat_a = st.number_input("纬度", value=st.session_state.point_a[0], format="%.6f")
        with col2:
            lon_a = st.number_input("经度", value=st.session_state.point_a[1], format="%.6f")
        if st.button("设置A点"):
            st.session_state.point_a = [lat_a, lon_a]
            st.rerun()
        
        # B点设置
        st.markdown("### 🎯 终点B")
        col1, col2 = st.columns(2)
        with col1:
            lat_b = st.number_input("纬度", value=st.session_state.point_b[0], format="%.6f", key="lat_b")
        with col2:
            lon_b = st.number_input("经度", value=st.session_state.point_b[1], format="%.6f", key="lon_b")
        if st.button("设置B点"):
            st.session_state.point_b = [lat_b, lon_b]
            st.rerun()
        
        st.markdown("---")
        
        # 飞行参数
        st.markdown("### ⚙️ 飞行参数")
        flight_alt = st.slider("飞行高度 (m)", 10, 150, st.session_state.flight_alt)
        st.session_state.flight_alt = flight_alt
        
        st.info("💡 **策略**\n• 高度<障碍物 → 绕行\n• 高度>障碍物 → 飞越")
        st.markdown("---")
        
        # 障碍物列表
        st.markdown("### 🚧 障碍物管理")
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 保存"):
                ObstaclePersistence.save_obstacles(st.session_state.obstacles)
                st.success("已保存")
        with col2:
            if st.button("📂 加载"):
                loaded, _ = ObstaclePersistence.load_obstacles()
                if loaded:
                    st.session_state.obstacles = loaded
                    st.rerun()
        with col3:
            if st.button("🗑️ 清空"):
                st.session_state.obstacles = []
                st.session_state.route_plans = []
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🗺️ 多航线规划")
        
        # 生成航线方案
        if st.button("🎯 生成航线方案", use_container_width=True, type="primary"):
            if st.session_state.point_a and st.session_state.point_b:
                start = st.session_state.point_a.copy()
                end = st.session_state.point_b.copy()
                if st.session_state.coord_type == "GCJ-02":
                    start = CoordConverter.gcj02_to_wgs84(start[0], start[1])
                    end = CoordConverter.gcj02_to_wgs84(end[0], end[1])
                planner = FlightPlanner(st.session_state.obstacles, flight_alt)
                plans = planner.generate_route_plans(start, end)
                if plans:
                    st.session_state.route_plans = plans
                    st.session_state.selected_plan = plans[0]
                    st.success(f"✅ 已生成 {len(plans)} 条航线")
                else:
                    st.error("无法生成航线")
                st.rerun()
            else:
                st.warning("请先设置A/B点")
        
        # 显示三个航线选项
        if st.session_state.route_plans:
            st.markdown("---")
            st.markdown("#### 📋 选择航线方案")
            
            for i, plan in enumerate(st.session_state.route_plans):
                # 三个不同颜色的卡片
                if "左绕行" in plan.name:
                    bg = "🟠"
                elif "右绕行" in plan.name:
                    bg = "🟣"
                else:
                    bg = "⭐"
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{bg} {plan.name}**")
                    st.caption(plan.description)
                with col2:
                    st.metric("距离", f"{plan.total_distance:.0f}m")
                with col3:
                    st.metric("时间", f"{plan.estimated_time:.0f}s")
                
                if st.button(f"选择此方案", key=f"select_{i}", use_container_width=True):
                    st.session_state.selected_plan = plan
                    st.success(f"已选择: {plan.name}")
                    st.rerun()
                st.markdown("---")
            
            # 确认使用航线
            if st.session_state.selected_plan:
                st.markdown("#### ✅ 当前选中")
                plan = st.session_state.selected_plan
                st.info(f"**{plan.name}** | {plan.total_distance:.0f}m | {plan.estimated_time:.0f}s")
                if st.button("✈️ 确认使用此航线", use_container_width=True, type="primary"):
                    st.session_state.flight_plan = {
                        'waypoints': plan.waypoints,
                        'total_distance': plan.total_distance,
                        'estimated_time': plan.estimated_time,
                        'path_type': plan.name,
                        'flight_altitude': flight_alt
                    }
                    st.success(f"✅ 已确认使用 {plan.name}")
                    st.rerun()


# ==================== 飞行监控页面 ====================
else:
    st.title("📡 飞行监控")
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🚁 飞行状态")
        
        if st.session_state.flight_plan:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("▶️ 开始飞行", type="primary") and not st.session_state.flying:
                    st.session_state.flying = True
                    st.session_state.sim = DroneSimulator(
                        st.session_state.flight_plan['waypoints'], 15,
                        st.session_state.flight_plan.get('flight_altitude', 50)
                    )
                    st.session_state.start_t = datetime.now()
                    st.session_state.alt_data = []
                    st.rerun()
            with col2:
                if st.button("⏸️ 暂停"):
                    st.session_state.flying = False
            with col3:
                if st.button("🛑 终止"):
                    st.session_state.flying = False
                    st.session_state.sim = None
        
        if st.session_state.flying and st.session_state.sim:
            status = st.session_state.sim.get_status()
            elapsed = (datetime.now() - st.session_state.start_t).total_seconds()
            hb = st.session_state.hb.send()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("当前航点", f"{status['current']}/{status['total']}")
            c2.metric("已用时间", f"{elapsed:.0f}s")
            c3.metric("剩余距离", f"{status['remaining']:.0f}m")
            c4.metric("进度", f"{status['progress']:.0f}%")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("电量", f"{max(0,100-elapsed/6):.0f}%")
            c2.metric("飞行高度", f"{status['altitude']:.0f}m")
            hb_stat = st.session_state.hb.get_status()
            c3.metric("心跳", f"{hb_stat['rate']}/min")
            c4.metric("延迟", f"{hb.get('delay',0)}ms")
            
            st.progress(int(status['progress']))
            
            st.session_state.alt_data.append({'t': elapsed, 'alt': status['altitude']})
            if len(st.session_state.alt_data) > 50:
                st.session_state.alt_data = st.session_state.alt_data[-50:]
            
            if len(st.session_state.alt_data) > 1:
                df = pd.DataFrame(st.session_state.alt_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['t'], y=df['alt'], mode='lines', name='高度(m)'))
                fig.update_layout(title="实时高度", xaxis_title="时间(秒)", yaxis_title="高度(m)")
                st.plotly_chart(fig, use_container_width=True)
            
            if status['progress'] >= 100:
                st.success("✅ 飞行完成！")
                st.session_state.flying = False
                st.balloons()
            else:
                st.session_state.sim.update(0.1)
                time.sleep(0.1)
                st.rerun()
        else:
            st.info("点击「开始飞行」启动监控")
    
    with col_right:
        st.subheader("💓 心跳监控")
        hb_stat = st.session_state.hb.get_status()
        
        c1, c2 = st.columns(2)
        c1.metric("发送", hb_stat['sent'])
        c2.metric("接收", hb_stat['recv'])
        status_text = "🟢 正常" if hb_stat['connected'] else "🔴 超时"
        st.metric("连接状态", status_text)
        
        st.markdown("---")
        st.markdown("### 📋 最新心跳")
        recent = st.session_state.hb.get_recent(8)
        if recent:
            df = pd.DataFrame(recent)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        if len(st.session_state.hb.recv_log) > 0:
            df_d = pd.DataFrame(st.session_state.hb.recv_log[-30:])
            fig = px.line(df_d, x='seq', y='delay', title="延迟趋势")
            st.plotly_chart(fig, use_container_width=True)
