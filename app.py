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
    """坐标系转换工具（WGS-84 ↔ GCJ-02）"""
    
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
    
    @classmethod
    def wgs84_to_gcj02(cls, lat, lon):
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
    """3D障碍物，包含高度信息"""
    def __init__(self, points: List[List[float]], min_height: float = 0, 
                 max_height: float = 100, name: str = "障碍物"):
        self.points = points
        self.min_height = min_height
        self.max_height = max_height
        self.name = name
        self.created_time = datetime.now()
        
        # 计算障碍物中心点
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
        self.width = max(lons) - min(lons)
        self.height = max(lats) - min(lats)
    
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
        if 'created_time' in data:
            obstacle.created_time = datetime.strptime(data['created_time'], "%Y-%m-%d %H:%M:%S")
        return obstacle
    
    def _point_in_polygon(self, point: List[float]) -> bool:
        x, y = point[1], point[0]
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i][1], self.points[i][0]
            x2, y2 = self.points[(i + 1) % n][1], self.points[(i + 1) % n][0]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def contains_point_2d(self, point: List[float]) -> bool:
        return self._point_in_polygon(point)
    
    def contains_point_3d(self, point: List[float], altitude: float) -> bool:
        if not self._point_in_polygon(point):
            return False
        return self.min_height <= altitude <= self.max_height
    
    def can_fly_over(self, altitude: float) -> bool:
        return altitude > self.max_height
    
    def can_fly_under(self, altitude: float) -> bool:
        return altitude < self.min_height
    
    def get_bypass_points(self, start: List[float], end: List[float], direction: str = "best") -> List[List[float]]:
        """获取绕行点
        direction: 'left' 左绕行, 'right' 右绕行, 'best' 最佳航线
        """
        # 计算从起点到终点的方向向量
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        # 垂直方向（左绕行和右绕行的方向）
        perp_x = -dy
        perp_y = dx
        
        # 计算偏移距离（基于障碍物大小）
        offset = max(self.width, self.height) * 0.25 + 0.002
        
        # 三个候选点
        left_point = [self.center[0] + perp_y * offset, self.center[1] + perp_x * offset]
        right_point = [self.center[0] - perp_y * offset, self.center[1] - perp_x * offset]
        
        # 计算最佳航线（取距离最短的）
        left_dist = self._calculate_path_length(start, left_point, end)
        right_dist = self._calculate_path_length(start, right_point, end)
        
        if left_dist <= right_dist:
            best_point = left_point
        else:
            best_point = right_point
        
        candidates = {
            'left': left_point,
            'right': right_point,
            'best': best_point
        }
        
        return candidates.get(direction, best_point)
    
    def _calculate_path_length(self, start, mid, end):
        """计算路径总长度"""
        return self._distance(start, mid) + self._distance(mid, end)
    
    def _distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def get_description(self) -> str:
        return f"""
        🚧 {self.name}
        📍 边界点数: {len(self.points)}
        📏 高度范围: {self.min_height}m - {self.max_height}m
        📐 尺寸: {self.width*111000:.0f}m x {self.height*111000:.0f}m
        """


# ==================== 障碍物持久化管理 ====================
class ObstaclePersistence:
    CONFIG_FILE = "obstacle_config.json"
    VERSION = "v16.0"
    
    @classmethod
    def save_obstacles(cls, obstacles: List[Obstacle3D]):
        config = {
            'version': cls.VERSION,
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
        except Exception as e:
            return [], None
    
    @classmethod
    def get_config_path(cls):
        return os.path.abspath(cls.CONFIG_FILE)
    
    @classmethod
    def get_config_status(cls):
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return {
                    'exists': True,
                    'count': config.get('count', 0),
                    'save_time': config.get('save_time', '未知'),
                    'version': config.get('version', '未知'),
                    'path': cls.get_config_path()
                }
            except:
                return {'exists': False, 'error': '读取失败'}
        return {'exists': False, 'count': 0}


# ==================== 航线规划方案类 ====================
class RoutePlan:
    """航线规划方案"""
    def __init__(self, name: str, waypoints: List[List[float]], total_distance: float, 
                 color: str, description: str, is_safe: bool = True):
        self.name = name
        self.waypoints = waypoints
        self.total_distance = total_distance
        self.estimated_time = total_distance / 15
        self.color = color
        self.description = description
        self.is_safe = is_safe
        self.num_waypoints = len(waypoints)


# ==================== 智能航线规划模块（多航线选择） ====================
class FlightPlanner:
    def __init__(self, obstacles: List[Obstacle3D], safe_radius: float, flight_altitude: float = 50):
        self.obstacles = obstacles
        self.safe_radius = safe_radius
        self.flight_altitude = flight_altitude
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        x, y = point[1], point[0]
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i][1], polygon[i][0]
            x2, y2 = polygon[(i + 1) % n][1], polygon[(i + 1) % n][0]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def line_intersects_obstacle(self, start: List[float], end: List[float], obstacle: Obstacle3D) -> bool:
        num_samples = 30
        for i in range(num_samples + 1):
            t = i / num_samples
            lat = start[0] + (end[0] - start[0]) * t
            lon = start[1] + (end[1] - start[1]) * t
            point = [lat, lon]
            if obstacle.contains_point_2d(point):
                return True
        return False
    
    def get_intersecting_obstacles(self, start: List[float], end: List[float]) -> List[Obstacle3D]:
        intersecting = []
        for obs in self.obstacles:
            if self.line_intersects_obstacle(start, end, obs):
                intersecting.append(obs)
        return intersecting
    
    def check_waypoint_safe(self, point: List[float]) -> bool:
        for obs in self.obstacles:
            if obs.contains_point_2d(point):
                return False
        return True
    
    def generate_route_plans(self, start: List[float], end: List[float]) -> List[RoutePlan]:
        """生成多个航线规划方案"""
        plans = []
        
        # 获取相交的障碍物
        intersecting = self.get_intersecting_obstacles(start, end)
        
        # 如果没有障碍物，返回直线路径
        if not intersecting:
            plans.append(RoutePlan(
                name="📏 直线路径",
                waypoints=[start, end],
                total_distance=self.calculate_distance(start, end),
                color="blue",
                description="✅ 无障碍物，直接飞行",
                is_safe=True
            ))
            return plans
        
        # 分类障碍物：需要绕行的
        need_bypass = []
        for obs in intersecting:
            if self.flight_altitude < obs.min_height:
                need_bypass.append(obs)
        
        if not need_bypass:
            # 可以飞越
            total_distance = self.calculate_distance(start, end)
            plans.append(RoutePlan(
                name="⬆️ 飞越路径",
                waypoints=[start, end],
                total_distance=total_distance,
                color="green",
                description=f"✅ 当前高度{self.flight_altitude}m可飞越",
                is_safe=True
            ))
            return plans
        
        # 需要绕行，生成三个方案
        first_obs = need_bypass[0]
        
        # 方案1: 左绕行
        left_point = first_obs.get_bypass_points(start, end, "left")
        if self.check_waypoint_safe(left_point):
            left_waypoints = [start, left_point, end]
            left_distance = self.calculate_distance(start, left_point) + self.calculate_distance(left_point, end)
            plans.append(RoutePlan(
                name="⬅️ 左绕行",
                waypoints=left_waypoints,
                total_distance=left_distance,
                color="orange",
                description=f"从「{first_obs.name}」左侧绕行",
                is_safe=True
            ))
        
        # 方案2: 右绕行
        right_point = first_obs.get_bypass_points(start, end, "right")
        if self.check_waypoint_safe(right_point):
            right_waypoints = [start, right_point, end]
            right_distance = self.calculate_distance(start, right_point) + self.calculate_distance(right_point, end)
            plans.append(RoutePlan(
                name="➡️ 右绕行",
                waypoints=right_waypoints,
                total_distance=right_distance,
                color="purple",
                description=f"从「{first_obs.name}」右侧绕行",
                is_safe=True
            ))
        
        # 方案3: 最佳航线（最短路径）
        best_point = first_obs.get_bypass_points(start, end, "best")
        if best_point and self.check_waypoint_safe(best_point):
            best_waypoints = [start, best_point, end]
            best_distance = self.calculate_distance(start, best_point) + self.calculate_distance(best_point, end)
            plans.append(RoutePlan(
                name="⭐ 最佳航线",
                waypoints=best_waypoints,
                total_distance=best_distance,
                color="gold",
                description=f"最短路径，节省{(max([p.total_distance for p in plans]) - best_distance) if plans else 0:.0f}m",
                is_safe=True
            ))
        
        # 按距离排序
        plans.sort(key=lambda x: x.total_distance)
        
        return plans


# ==================== 无人机模拟器模块 ====================
class DroneSimulator:
    def __init__(self, waypoints: List[List[float]], speed: float = 15, altitude: float = 50):
        self.waypoints = waypoints
        self.speed = speed
        self.altitude = altitude
        self.current_waypoint_index = 0
        self.current_position = waypoints[0].copy() if waypoints else [0, 0]
        self.completed_distance = 0.0
        self.total_distance = self._calculate_total_distance()
        self.is_flying = True
    
    def _calculate_distance(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        R = 6371000
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def _calculate_total_distance(self):
        total = 0
        for i in range(len(self.waypoints) - 1):
            total += self._calculate_distance(self.waypoints[i], self.waypoints[i+1])
        return total
    
    def update(self, delta_time=0.1):
        if not self.is_flying or self.current_waypoint_index >= len(self.waypoints) - 1:
            return False
        
        target = self.waypoints[self.current_waypoint_index + 1]
        distance_to_target = self._calculate_distance(self.current_position, target)
        step_distance = self.speed * delta_time
        
        if distance_to_target <= step_distance:
            self.current_position = target
            self.completed_distance += distance_to_target
            self.current_waypoint_index += 1
        else:
            ratio = step_distance / distance_to_target
            new_lat = self.current_position[0] + (target[0] - self.current_position[0]) * ratio
            new_lon = self.current_position[1] + (target[1] - self.current_position[1]) * ratio
            self.current_position = [new_lat, new_lon]
            self.completed_distance += step_distance
        
        return self.current_waypoint_index < len(self.waypoints) - 1
    
    def get_status(self):
        progress = (self.completed_distance / self.total_distance * 100) if self.total_distance > 0 else 0
        return {
            'position': self.current_position,
            'current_waypoint': self.current_waypoint_index + 1,
            'total_waypoints': len(self.waypoints),
            'remaining_distance': self.total_distance - self.completed_distance,
            'progress': progress,
            'completed_distance': self.completed_distance,
            'altitude': self.altitude
        }


# ==================== 心跳监控模块 ====================
class HeartbeatMonitor:
    def __init__(self):
        self.sequence_number = 0
        self.send_log = []
        self.receive_log = []
        self.timeout_log = []
        self.last_heartbeat_time = None
        self.is_connected = False
    
    def send_heartbeat(self):
        self.sequence_number += 1
        send_time = datetime.now()
        heartbeat = {'seq': self.sequence_number, 'send_time': send_time}
        self.send_log.append(heartbeat)
        
        delay_ms = random.uniform(5, 50)
        receive_time = send_time + timedelta(milliseconds=delay_ms)
        heartbeat['receive_time'] = receive_time
        heartbeat['delay_ms'] = round(delay_ms, 2)
        self.receive_log.append(heartbeat)
        self.last_heartbeat_time = receive_time
        self.is_connected = True
        return heartbeat
    
    def check_timeout(self):
        current_time = datetime.now()
        if self.last_heartbeat_time:
            elapsed = (current_time - self.last_heartbeat_time).total_seconds()
            if elapsed > 3:
                self.is_connected = False
                self.timeout_log.append({
                    'time': current_time,
                    'elapsed': elapsed,
                    'message': f'连接超时: {elapsed:.1f}秒未收到心跳'
                })
        return self.is_connected
    
    def get_status(self):
        self.check_timeout()
        total_sent = len(self.send_log)
        total_received = len(self.receive_log)
        return {
            'heartbeat_rate': 60 if self.is_connected else 0,
            'last_heartbeat_time': self.last_heartbeat_time,
            'total_sent': total_sent,
            'total_received': total_received,
            'is_connected': self.is_connected,
            'success_rate': (total_received / total_sent * 100) if total_sent > 0 else 0,
            'timeout_count': len(self.timeout_log)
        }
    
    def get_recent_heartbeats(self, n=10):
        recent = []
        for h in self.receive_log[-n:]:
            recent.append({
                'seq': h['seq'],
                'send_time': h['send_time'].strftime("%H:%M:%S.%f")[:-3],
                'receive_time': h['receive_time'].strftime("%H:%M:%S.%f")[:-3],
                'delay_ms': h['delay_ms']
            })
        return recent


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="南京科技职业学院 - 无人机智能监控系统",
    page_icon="🛰️",
    layout="wide"
)

# 南京科技职业学院坐标（GCJ-02）
CAMPUS_CENTER = [32.234097, 118.749413]

# 初始化session state
if 'page' not in st.session_state:
    st.session_state.page = "航线规划"
if 'obstacles' not in st.session_state:
    saved_obstacles, config = ObstaclePersistence.load_obstacles()
    st.session_state.obstacles = saved_obstacles if saved_obstacles else []
if 'temp_obstacle' not in st.session_state:
    st.session_state.temp_obstacle = None
if 'temp_obstacle_height' not in st.session_state:
    st.session_state.temp_obstacle_height = [0, 30]
if 'temp_obstacle_name' not in st.session_state:
    st.session_state.temp_obstacle_name = "建筑物"
if 'route_plans' not in st.session_state:
    st.session_state.route_plans = []
if 'selected_plan_index' not in st.session_state:
    st.session_state.selected_plan_index = 0
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None
if 'coord_type' not in st.session_state:
    st.session_state.coord_type = "GCJ-02"
if 'point_a' not in st.session_state:
    st.session_state.point_a = [32.2323, 118.749]
if 'point_b' not in st.session_state:
    st.session_state.point_b = [32.2344, 118.749]
if 'heartbeat_monitor' not in st.session_state:
    st.session_state.heartbeat_monitor = HeartbeatMonitor()
if 'is_flying' not in st.session_state:
    st.session_state.is_flying = False
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'altitude_data' not in st.session_state:
    st.session_state.altitude_data = []
if 'flight_altitude' not in st.session_state:
    st.session_state.flight_altitude = 20


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🛰️ 无人机系统")
    st.caption("南京科技职业学院 · 智能监控平台")
    st.markdown("---")
    
    st.subheader("📱 功能页面")
    page = st.radio("", ["🗺️ 航线规划", "📡 飞行监控"], label_visibility="collapsed")
    st.session_state.page = page
    
    st.markdown("---")
    
    st.subheader("🌐 坐标系设置")
    coord_type = st.selectbox("输入坐标系", ["WGS-84", "GCJ-02 (高德/百度)"])
    st.session_state.coord_type = "WGS-84" if coord_type == "WGS-84" else "GCJ-02"
    
    if st.session_state.coord_type == "GCJ-02":
        st.info("📍 当前使用 GCJ-02 坐标系\n(高德/百度地图)")
    else:
        st.info("🌍 当前使用 WGS-84 坐标系\n(GPS/国际标准)")
    
    st.markdown("---")
    
    st.subheader("📊 系统状态")
    st.success("✅ 系统运行正常")
    
    config_status = ObstaclePersistence.get_config_status()
    if config_status['exists']:
        st.info(f"💾 障碍物配置\n共 {config_status['count']} 个 | {config_status['save_time']}")


# ==================== 页面1: 航线规划 ====================
if st.session_state.page == "🗺️ 航线规划":
    st.title("🛰️ 航线规划")
    st.markdown("设置起降点、3D障碍区，**多航线选择**（左绕行/右绕行/最佳航线）— **南京科技职业学院**")
    st.markdown("---")
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.subheader("🛰️ 卫星地图")
        st.caption("📍 南京科技职业学院 | 坐标: 32.2341°N, 118.7494°E")
        
        m = folium.Map(location=CAMPUS_CENTER, zoom_start=18, control_scale=True)
        
        # 高德卫星图
        folium.TileLayer(
            tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
            attr='高德卫星地图',
            subdomains=['1', '2', '3', '4'],
            name='卫星地图'
        ).add_to(m)
        
        # OpenStreetMap
        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr='OpenStreetMap',
            name='街道地图'
        ).add_to(m)
        
        # 学院标注
        folium.Marker(
            CAMPUS_CENTER,
            popup=folium.Popup('<b>🏫 南京科技职业学院</b><br>南京市江北新区欣乐路188号', max_width=250),
            icon=folium.Icon(color='red', icon='university', prefix='fa')
        ).add_to(m)
        
        # 绘制障碍区
        for i, obstacle in enumerate(st.session_state.obstacles):
            if len(obstacle.points) >= 3:
                display_obs = obstacle.points
                if st.session_state.coord_type == "GCJ-02":
                    display_obs = []
                    for p in obstacle.points:
                        wgs_lat, wgs_lon = CoordConverter.gcj02_to_wgs84(p[0], p[1])
                        display_obs.append([wgs_lat, wgs_lon])
                
                # 根据高度选择颜色
                flight_h = st.session_state.flight_altitude
                if flight_h < obstacle.min_height:
                    color = 'orange'
                    fill_opacity = 0.3
                elif flight_h > obstacle.max_height:
                    color = 'green'
                    fill_opacity = 0.2
                else:
                    color = 'red'
                    fill_opacity = 0.5
                
                folium.Polygon(
                    locations=[[p[0], p[1]] for p in display_obs],
                    color=color, weight=2, fill=True, fill_color=color, fill_opacity=fill_opacity,
                    popup=folium.Popup(obstacle.get_description(), max_width=200)
                ).add_to(m)
        
        # 绘制A点和B点
        if st.session_state.point_a:
            display_a = st.session_state.point_a
            if st.session_state.coord_type == "GCJ-02":
                display_a = CoordConverter.gcj02_to_wgs84(display_a[0], display_a[1])
            folium.Marker([display_a[0], display_a[1]], popup='🚁 起飞点 A',
                         icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
        
        if st.session_state.point_b:
            display_b = st.session_state.point_b
            if st.session_state.coord_type == "GCJ-02":
                display_b = CoordConverter.gcj02_to_wgs84(display_b[0], display_b[1])
            folium.Marker([display_b[0], display_b[1]], popup='🎯 目标点 B',
                         icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')).add_to(m)
        
        # 绘制所有航线方案
        if st.session_state.route_plans:
            for i, plan in enumerate(st.session_state.route_plans):
                display_wps = []
                for wp in plan.waypoints:
                    if st.session_state.coord_type == "GCJ-02":
                        wp = CoordConverter.gcj02_to_wgs84(wp[0], wp[1])
                    display_wps.append([wp[0], wp[1]])
                
                # 高亮选中的方案
                if i == st.session_state.selected_plan_index:
                    line_width = 4
                    line_opacity = 0.9
                else:
                    line_width = 2
                    line_opacity = 0.4
                
                folium.PolyLine(
                    display_wps, 
                    color=plan.color, 
                    weight=line_width, 
                    opacity=line_opacity,
                    popup=f"{plan.name}\n{plan.description}\n距离: {plan.total_distance:.0f}m"
                ).add_to(m)
                
                # 添加绕行点标记
                if len(display_wps) > 2:
                    for wp in display_wps[1:-1]:
                        folium.Marker(
                            wp, 
                            popup=f'绕行点',
                            icon=folium.Icon(color='purple', icon='refresh', prefix='fa', icon_size=(20, 20))
                        ).add_to(m)
        
        # 绘图工具
        draw = plugins.Draw(draw_options={
            'polyline': False, 'rectangle': False, 'circle': False,
            'marker': True, 'polygon': {'allowIntersection': False}, 'circlemarker': False
        }, edit_options={'edit': True})
        draw.add_to(m)
        
        plugins.MeasureControl(position='topleft').add_to(m)
        plugins.Fullscreen().add_to(m)
        folium.LayerControl().add_to(m)
        
        output = st_folium(m, width=750, height=550, key="planning_map")
        
        if output and 'last_active_drawing' in output:
            drawing = output['last_active_drawing']
            if drawing and drawing['geometry']['type'] == 'Polygon':
                coords = drawing['geometry']['coordinates'][0]
                points = [[c[1], c[0]] for c in coords]
                if len(points) >= 3:
                    st.session_state.temp_obstacle = points
                    st.success(f"✅ 已绘制 {len(points)} 个点的障碍区，请在右侧设置高度和名称")
    
    with col_right:
        st.subheader("🎯 控制面板")
        
        # 新建障碍物
        if st.session_state.temp_obstacle:
            st.markdown("### 🆕 新建3D障碍物")
            st.caption(f"已绘制 {len(st.session_state.temp_obstacle)} 个边界点")
            
            obs_name = st.text_input("障碍物名称", value=st.session_state.temp_obstacle_name,
                                      placeholder="例如：教学楼、图书馆")
            st.session_state.temp_obstacle_name = obs_name if obs_name else "障碍物"
            
            st.markdown("**📏 高度范围设置**")
            col1, col2 = st.columns(2)
            with col1:
                min_h = st.number_input("最低高度 (m)", value=st.session_state.temp_obstacle_height[0],
                                        min_value=0, max_value=200, step=5,
                                        help="障碍物底部离地高度")
            with col2:
                max_h = st.number_input("最高高度 (m)", value=st.session_state.temp_obstacle_height[1],
                                        min_value=min_h + 1, max_value=300, step=5,
                                        help="障碍物顶部高度")
            st.session_state.temp_obstacle_height = [min_h, max_h]
            
            st.caption(f"📊 障碍物高度: {min_h}m - {max_h}m")
            progress_value = min(1.0, max(0.0, max_h / 300))
            st.progress(progress_value)
            
            current_flight_h = st.session_state.flight_altitude
            if current_flight_h < min_h:
                st.success(f"💡 当前飞行高度 {current_flight_h}m < 障碍物底部，将自动绕行")
            elif current_flight_h > max_h:
                st.success(f"💡 当前飞行高度 {current_flight_h}m > 障碍物顶部，可直接飞越")
            else:
                st.warning(f"⚠️ 当前飞行高度与障碍物冲突，建议调整")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 保存障碍物", use_container_width=True, type="primary"):
                    new_obstacle = Obstacle3D(st.session_state.temp_obstacle, min_h, max_h, 
                                              st.session_state.temp_obstacle_name)
                    st.session_state.obstacles.append(new_obstacle)
                    st.session_state.temp_obstacle = None
                    st.session_state.temp_obstacle_height = [0, 30]
                    st.session_state.temp_obstacle_name = "建筑物"
                    st.success(f"✅ 已添加3D障碍物")
                    st.rerun()
            with col2:
                if st.button("🗑️ 取消", use_container_width=True):
                    st.session_state.temp_obstacle = None
                    st.rerun()
            st.markdown("---")
        
        # 定位按钮
        st.markdown("### 🏫 校园快速定位")
        if st.button("📍 定位南京科技职业学院", use_container_width=True):
            st.success("已定位到学院中心")
            st.rerun()
        st.markdown("---")
        
        # A点设置
        st.markdown("### 🚁 起点 A")
        st.caption(f"输入坐标系: {st.session_state.coord_type}")
        col1, col2 = st.columns(2)
        with col1:
            lat_a = st.number_input("纬度", value=st.session_state.point_a[0], format="%.6f", key="lat_a")
        with col2:
            lon_a = st.number_input("经度", value=st.session_state.point_a[1], format="%.6f", key="lon_a")
        if st.button("📍 设置 A 点", use_container_width=True):
            st.session_state.point_a = [lat_a, lon_a]
            st.success(f"起点已设置: ({lat_a}, {lon_a})")
            st.rerun()
        
        # B点设置
        st.markdown("### 🎯 终点 B")
        col1, col2 = st.columns(2)
        with col1:
            lat_b = st.number_input("纬度", value=st.session_state.point_b[0], format="%.6f", key="lat_b")
        with col2:
            lon_b = st.number_input("经度", value=st.session_state.point_b[1], format="%.6f", key="lon_b")
        if st.button("🏁 设置 B 点", use_container_width=True):
            st.session_state.point_b = [lat_b, lon_b]
            st.success(f"终点已设置: ({lat_b}, {lon_b})")
            st.rerun()
        
        st.markdown("---")
        
        # 飞行参数
        st.markdown("### ⚙️ 飞行参数")
        flight_altitude = st.slider("设定飞行高度 (m)", 10, 150, st.session_state.flight_altitude,
                                     help="低于障碍物时自动绕行，高于障碍物时直接飞越")
        st.session_state.flight_altitude = flight_altitude
        safe_radius = st.slider("安全半径 (m)", 10, 100, 30)
        
        # 智能避障说明
        st.info("💡 **多航线选择**\n\n"
                "• ⬅️ 左绕行 - 从障碍物左侧绕过\n"
                "• ➡️ 右绕行 - 从障碍物右侧绕过\n"
                "• ⭐ 最佳航线 - 自动选择最短路径")
        
        st.markdown("---")
        
        # 障碍物管理
        st.markdown("### 🚧 障碍物管理")
        st.caption(f"📦 共 {len(st.session_state.obstacles)} 个3D障碍物")
        
        if st.session_state.obstacles:
            for i, obs in enumerate(st.session_state.obstacles):
                if flight_altitude < obs.min_height:
                    status_icon = "🔄"
                    status_text = "可绕行"
                elif flight_altitude > obs.max_height:
                    status_icon = "⬆️"
                    status_text = "可飞越"
                else:
                    status_icon = "⚠️"
                    status_text = "冲突"
                    
                with st.expander(f"{status_icon} {obs.name} ({len(obs.points)}点) - {status_text}"):
                    st.caption(f"📏 高度范围: {obs.min_height}m - {obs.max_height}m")
                    if flight_altitude < obs.min_height:
                        st.success(f"✅ 飞行高度 {flight_altitude}m < 障碍物底部，将自动绕行")
                    elif flight_altitude > obs.max_height:
                        st.success(f"✅ 飞行高度 {flight_altitude}m > 障碍物顶部，可直接飞越")
                    else:
                        st.warning(f"⚠️ 飞行高度 {flight_altitude}m 与障碍物高度范围冲突！")
                    if st.button(f"🗑️ 删除", key=f"del_{i}"):
                        st.session_state.obstacles.pop(i)
                        st.rerun()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 保存配置", use_container_width=True):
                success, _ = ObstaclePersistence.save_obstacles(st.session_state.obstacles)
                if success:
                    st.success(f"✅ 已保存 {len(st.session_state.obstacles)} 个3D障碍物")
                else:
                    st.error("保存失败")
        with col2:
            if st.button("📂 加载配置", use_container_width=True):
                loaded, _ = ObstaclePersistence.load_obstacles()
                if loaded:
                    st.session_state.obstacles = loaded
                    st.success(f"✅ 已加载 {len(loaded)} 个3D障碍物")
                    st.rerun()
                else:
                    st.warning("没有找到保存的配置")
        with col3:
            if st.button("🗑️ 清除全部", use_container_width=True):
                st.session_state.obstacles = []
                st.success("已清除所有障碍物")
                st.rerun()
        
        # 下载配置
        st.markdown("---")
        st.markdown("### 📥 下载配置文件")
        config_status = ObstaclePersistence.get_config_status()
        if config_status['exists']:
            with open(ObstaclePersistence.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_content = f.read()
            st.download_button(label="📥 下载 obstacle_config.json", data=config_content,
                               file_name="obstacle_config.json", mime="application/json", use_container_width=True)
        
        st.markdown("---")
        
        # ========== 多航线选择区域 ==========
        st.markdown("### 🗺️ 多航线规划")
        
        # 生成航线方案按钮
        if st.button("🎯 生成航线方案", use_container_width=True, type="primary"):
            if st.session_state.point_a and st.session_state.point_b:
                start = st.session_state.point_a.copy()
                end = st.session_state.point_b.copy()
                if st.session_state.coord_type == "GCJ-02":
                    start = CoordConverter.gcj02_to_wgs84(start[0], start[1])
                    end = CoordConverter.gcj02_to_wgs84(end[0], end[1])
                
                planner = FlightPlanner(st.session_state.obstacles, safe_radius, flight_altitude)
                plans = planner.generate_route_plans(list(start), list(end))
                
                if plans:
                    st.session_state.route_plans = plans
                    st.session_state.selected_plan_index = 0
                    st.session_state.selected_plan = plans[0]
                    st.success(f"✅ 已生成 {len(plans)} 条航线方案")
                else:
                    st.error("❌ 无法生成航线方案")
                    st.session_state.route_plans = []
                    st.session_state.selected_plan = None
                st.rerun()
            else:
                st.warning("请先设置 A 点和 B 点")
        
        # 显示航线方案选择
        if st.session_state.route_plans:
            st.markdown("---")
            st.markdown("#### 📋 可选航线方案")
            
            # 用卡片形式显示三个方案
            for i, plan in enumerate(st.session_state.route_plans):
                # 根据方案类型设置背景色
                if "左绕行" in plan.name:
               
