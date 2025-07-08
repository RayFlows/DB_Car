import socket
import threading
import json
import time
from flask import current_app

# 全局存储雷达数据
radar_data = {"Left": 0.0, "Right": 0.0, "Rear": 0.0}
last_update_time = 0
lock = threading.Lock()

def get_radar_data():
    """获取最新的雷达数据"""
    with lock:
        return radar_data.copy()

def radar_server():
    """启动雷达数据接收服务器"""
    HOST = '0.0.0.0'  # 监听所有接口
    PORT = 5557       # 监听端口
    
    print(f"[RADAR SERVER] Starting radar data server on port {PORT}")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(1.0)
        
        print(f"[RADAR SERVER] Listening on {HOST}:{PORT}. Waiting for connections...")
        
        while True:
            try:
                conn, addr = s.accept()
                print(f"[RADAR SERVER] New connection from {addr[0]}:{addr[1]}")
                
                # 处理客户端连接
                with conn:
                    client_ip = addr[0]
                    print(f"[RADAR SERVER] Handling connection from {client_ip}")
                    
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                print(f"[RADAR SERVER] Client {client_ip} disconnected")
                                break
                            
                            # 打印接收到的原始数据
                            print(f"[RADAR SERVER] Received raw data from {client_ip}: {data}")
                            
                            # 尝试解码数据
                            try:
                                decoded_data = data.decode('utf-8')
                                print(f"[RADAR SERVER] Decoded data: {decoded_data}")
                            except UnicodeDecodeError:
                                print(f"[RADAR SERVER] Failed to decode data from {client_ip}")
                                continue
                            
                            # 尝试解析JSON数据
                            try:
                                message = json.loads(decoded_data)
                                print(f"[RADAR SERVER] Parsed JSON message: {message}")
                                
                                if message.get('type') == 'radar':
                                    with lock:
                                        radar_data.update(message['data'])
                                        last_update_time = time.time()
                                        print(f"[RADAR SERVER] Updated radar data: {radar_data}")
                                        print(f"[RADAR SERVER] Left: {radar_data['Left']:.1f} cm, "
                                              f"Right: {radar_data['Right']:.1f} cm, "
                                              f"Rear: {radar_data['Rear']:.1f} cm")
                                
                                elif message.get('heartbeat') == 'ping':
                                    print(f"[RADAR SERVER] Received heartbeat from {client_ip}")
                                    # 心跳包，不做处理
                                else:
                                    print(f"[RADAR SERVER] Unknown message type: {message}")
                            
                            except json.JSONDecodeError as e:
                                print(f"[RADAR SERVER] Invalid JSON data from {client_ip}: {e}")
                                print(f"[RADAR SERVER] Problematic data: {decoded_data}")
                        
                        except ConnectionResetError:
                            print(f"[RADAR SERVER] Connection reset by client {client_ip}")
                            break
                        except Exception as e:
                            print(f"[RADAR SERVER] Error handling client {client_ip}: {e}")
                            break
            except socket.timeout:
                # 正常超时，继续等待新连接
                pass
            except Exception as e:
                print(f"[RADAR SERVER] Error in server loop: {e}")
                time.sleep(1)

def start_radar_server():
    """启动雷达服务器线程"""
    print("[RADAR SERVER] Starting radar server thread...")
    server_thread = threading.Thread(target=radar_server, daemon=True)
    server_thread.start()
    print("[RADAR SERVER] Radar server thread started")