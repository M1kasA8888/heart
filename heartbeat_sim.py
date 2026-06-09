import time
import threading
import queue
from datetime import datetime
import random

class DroneHeartbeatSimulator:
    """无人机心跳模拟器"""
    
    def __init__(self):
        self.running = False
        self.heartbeat_queue = queue.Queue()
        self.last_heartbeat_time = None
        self.offline = False
        self.heartbeat_history = []  # 存储历史数据
        
    def start(self):
        """启动模拟器"""
        self.running = True
        self.offline = False
        self.heartbeat_history = []
        
        # 启动心跳发送线程
        self.send_thread = threading.Thread(target=self._send_heartbeat)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        # 启动掉线检测线程
        self.detect_thread = threading.Thread(target=self._detect_offline)
        self.detect_thread.daemon = True
        self.detect_thread.start()
        
    def _send_heartbeat(self):
        """模拟发送心跳信号"""
        while self.running:
            # 随机模拟呼吸时间（0.8-1.2秒之间变化，更真实）
            breath_time = round(random.uniform(0.8, 1.2), 2)
            
            heartbeat = {
                'timestamp': datetime.now(),
                'time_str': datetime.now().strftime("%H:%M:%S"),
                'breath_time': breath_time,
                'heartbeat_id': len(self.heartbeat_history) + 1,
                'status': 'alive'
            }
            
            self.heartbeat_queue.put(heartbeat)
            self.last_heartbeat_time = time.time()
            self.heartbeat_history.append(heartbeat)
            
            # 保留最近100条记录
            if len(self.heartbeat_history) > 100:
                self.heartbeat_history.pop(0)
                
            time.sleep(1)  # 每秒发送一次
            
    def _detect_offline(self):
        """掉线检测：3秒没收到就报警"""
        while self.running:
            if self.last_heartbeat_time is not None:
                elapsed = time.time() - self.last_heartbeat_time
                if elapsed > 3 and not self.offline:
                    self.offline = True
                    # 添加掉线标记
                    offline_mark = {
                        'timestamp': datetime.now(),
                        'time_str': datetime.now().strftime("%H:%M:%S"),
                        'breath_time': 0,
                        'heartbeat_id': len(self.heartbeat_history) + 1,
                        'status': 'offline',
                        'offline_after': round(elapsed, 1)
                    }
                    self.heartbeat_queue.put(offline_mark)
                    self.heartbeat_history.append(offline_mark)
                    
            time.sleep(0.5)
            
    def stop(self):
        """停止模拟器"""
        self.running = False
        
    def get_latest_heartbeat(self):
        """获取最新的心跳数据"""
        try:
            return self.heartbeat_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_history(self):
        """获取历史数据"""
        return self.heartbeat_history.copy()
