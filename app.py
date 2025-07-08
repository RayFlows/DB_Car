# app.py

from flask import Flask, render_template, Response
import threading
import camera_receiver
import video_classification
import time
import requests
from radar_server import start_radar_server, get_radar_data  # 新增

app = Flask(__name__)

# 全局状态变量
current_cat_breed = "Waiting..."
current_confidence = 0.0

# 启动雷达服务器
start_radar_server()

# 允许显示的控制按键列表
ALLOWED_KEYS = {
    'w', 'a', 's', 'd', 'q', 'e', 'x', 
    '+', '-', '1', '2', '3', 
    'left', 'right', 'up', 'down', 'space', '0',
    'c', 'v', 'b', 'n', 'm'
}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(video_classification.generate_processed_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/status')
# def get_status():
#     """获取状态信息"""
#     return {
#         'cat_breed': current_cat_breed,
#         'confidence': current_confidence
#     }
@app.route('/status')
def get_status():
    """获取状态信息"""
    # 从键盘控制器获取按键状态
    try:
        response = requests.get('http://localhost:5001/keys', timeout=0.1)
        if response.status_code == 200:
            active_keys = response.json()['keys']
        else:
            active_keys = []
    except:
        active_keys = []

    # 过滤只显示允许的控制按键
    filtered_keys = [key for key in active_keys if key in ALLOWED_KEYS]

    # 获取雷达数据
    radar_info = get_radar_data()

    # 当置信度低于0.7时，显示"分析中..."而不是品种名称
    display_breed = current_cat_breed if current_confidence >= 0.7 else "分析中..."

    return {
        'keys': filtered_keys,#active_keys,
        'cat_breed': display_breed,#current_cat_breed,
        'confidence': current_confidence,
        'radar': radar_info  # 包含雷达数据
    }

def update_status():
    """更新状态信息"""
    global current_cat_breed, current_confidence
    while True:
        breed, conf = video_classification.get_last_prediction()
        if breed:
            current_cat_breed = breed
            current_confidence = conf
        time.sleep(0.5)  # 不需要高频率更新

def start_background_tasks():
    """启动后台任务"""
    # 启动视频接收
    video_thread = threading.Thread(target=camera_receiver.run, daemon=True)
    video_thread.start()
    
    # 启动状态更新
    status_thread = threading.Thread(target=update_status, daemon=True)
    status_thread.start()

if __name__ == '__main__':
    # 初始化模型
    video_classification.initialize_models()
    
    # 启动后台任务
    start_background_tasks()
    
    # 启动Flask服务器（低优先级）
    app.run(host='0.0.0.0', port=5000, threaded=True)