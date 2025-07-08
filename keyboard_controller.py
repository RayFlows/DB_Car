#keyboard_controller.py
import keyboard
import socket
import time
import threading
import sys
from flask import Flask, jsonify

# 配置信息
RASPBERRY_IP = "172.20.10.2"   #"frp-fit.com"  # 树莓派公网IP
CONTROL_PORT = 5556            #26669           # 树莓派控制端口



# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.1)  # 设置超时时间
print(f"UDP套接字已创建，目标: {RASPBERRY_IP}:{CONTROL_PORT}")

# 命令映射
KEY_COMMANDS = {
    'w': 'w',  # 前进
    'a': 'a',  # 左转
    's': 's',  # 后退
    'd': 'd',  # 右转
    'q': 'q',  # 斜左前
    'e': 'e',  # 斜右前
    'x': 'x',  # 停止
    
    '+': '+',   # 加速
    '-': '-', # 减速
    '1': '1',    # 速度1
    '2': '2',    # 速度2
    '3': '3',    # 速度3
    
    'left': '4',    # 舵机左转
    'right': '6',   # 舵机右转
    'up': '7',      # 舵机上转
    'down': '9',    # 舵机下转
    'space': '5',   # 舵机停止
    '0': '8',       # 舵机复位
    
    'c': 'c',  # 显示猫品种1
    'v': 'v',  # 显示猫品种2
    'b': 'b',  # 显示猫品种3
    'n': 'n',  # 显示猫品种4
    'm': 'm',  # 显示猫品种5
}

# 当前活动命令和发送状态
active_commands = {}
send_thread = None
running = True
server_started = False  # 服务器启动状态标志

# 当前活动按键状态
active_keys = set()

# 创建Flask应用
keyboard_app = Flask(__name__)

@keyboard_app.route('/keys')
def get_active_keys():
    """获取当前按下的键"""
    return jsonify({"keys": list(active_keys)})

# def run_keyboard_server():
#     """运行键盘状态服务器"""
#     keyboard_app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)

def run_keyboard_server():
    """运行键盘状态服务器"""
    global server_started
    try:
        print("正在启动键盘状态服务器...")
        # 使用不同的端口避免冲突
        keyboard_app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)
        server_started = True
    except Exception as e:
        print(f"键盘状态服务器启动失败: {e}")
        # 尝试备用端口
        try:
            print("尝试使用备用端口 5002...")
            keyboard_app.run(host='0.0.0.0', port=5002, threaded=True, use_reloader=False)
            server_started = True
        except Exception as e2:
            print(f"备用端口启动失败: {e2}")
            server_started = False

def send_command(cmd):
    """发送命令到树莓派"""
    try:
        print(f"尝试发送命令: {cmd}")
        sock.sendto(cmd.encode(), (RASPBERRY_IP, CONTROL_PORT))
        print(f"已发送: {cmd} 到 {RASPBERRY_IP}:{CONTROL_PORT}")
    except Exception as e:
        print(f"发送失败: {e}")

def command_sender():
    """持续发送活动命令的线程"""
    print("命令发送线程已启动")
    while running:
        try:
            for cmd in list(active_commands.keys()):
                if active_commands[cmd]:
                    send_command(cmd)
            time.sleep(0.05)  # 控制发送频率(20Hz)
        except Exception as e:
            print(f"发送线程错误: {e}")
    print("命令发送线程已停止")

def on_key_event(e):
    """处理键盘事件"""
    global send_thread
    
    print(f"键盘事件: {e.event_type} 键: {e.name}")
    
    if e.event_type == keyboard.KEY_DOWN:
        # 添加到活动键集合
        active_keys.add(e.name)

        if e.name in KEY_COMMANDS:
            cmd = KEY_COMMANDS[e.name]
            print(f"检测到按键按下: {e.name} -> 命令: {cmd}")
            active_commands[cmd] = True
            
            # 对于非持续运动命令，立即发送一次
            if cmd in ['+', '-', '1', '2', '3', '4', '5', '6', '8', 'c', 'v', 'b', 'n', 'm']:
                send_command(cmd)
                active_commands[cmd] = False  # 立即标记为非活动
            
            # 启动发送线程（如果未启动）
            if send_thread is None:
                send_thread = threading.Thread(target=command_sender, daemon=True)
                send_thread.start()
                print("启动命令发送线程")
    
    elif e.event_type == keyboard.KEY_UP:
        # 从活动键集合中移除
        if e.name in active_keys:
            active_keys.remove(e.name)
            
        if e.name in KEY_COMMANDS:
            cmd = KEY_COMMANDS[e.name]
            print(f"检测到按键释放: {e.name} -> 命令: {cmd}")
            if cmd in active_commands:
                active_commands[cmd] = False
                
                # 对于运动命令，发送停止命令
                if cmd in ['w', 'a', 's', 'd', 'q', 'e']:
                    send_command('x')

def check_port_available(port):
    """检查端口是否可用"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("localhost", port))
        s.close()
        return True
    except:
        return False

def main():
    global server_started

    print("="*50)
    print("猫咪迷宫遥控器 - 键盘控制模式")
    print("="*50)
    print("方向控制: W(前), A(左), S(后), D(右), Q(斜左前), E(斜右前)")
    print("速度控制: ↑(加速), ↓(减速), 1/2/3(挡位)")
    print("舵机控制: ←(左转), →(右转), 空格(停舵), 0(复位)")
    print("猫品种显示: C/V/B/N/M")
    print("退出: ESC")

    # 检查端口可用性
    port = 5001
    if not check_port_available(port):
        print(f"⚠️ 警告：端口 {port} 已被占用，尝试使用备用端口 5002")
        port = 5002
        if not check_port_available(port):
            print(f"❌ 错误：端口 {port} 也被占用，请关闭占用程序或选择其他端口")
            port = int(input("请输入可用端口号: "))

    # 启动键盘状态服务器
    print(f"\n启动键盘状态服务器在端口 {port}...")
    keyboard_app.config['SERVER_PORT'] = port
    
    kb_server_thread = threading.Thread(target=run_keyboard_server, daemon=True)
    kb_server_thread.start()

    # 等待服务器启动
    time.sleep(5)  # 给服务器启动时间
    if server_started:
        print(f"✅ 键盘状态服务器运行在 http://localhost:{port}/keys")
        print(f"您可以在浏览器中访问此URL查看当前按下的键")
    else:
        print("❌ 键盘状态服务器启动失败，状态API将不可用")

    # 测试网络连接
    print("\n测试网络连接...")
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_sock.sendto(b"TEST", (RASPBERRY_IP, CONTROL_PORT))
        print(f"测试数据包已发送到 {RASPBERRY_IP}:{CONTROL_PORT}")
    except Exception as e:
        print(f"网络测试失败: {e}")
        print("请检查: ")
        print("1. 树莓派是否在线并运行控制服务器")
        print("2. 防火墙设置是否允许UDP出站")
        print("3. 公网地址是否正确")
    
    # 设置键盘监听
    print("\n启动键盘监听...")
    keyboard.hook(on_key_event)
    print("键盘监听已启动")
    
    # 等待退出
    print("\n等待键盘输入 (按ESC退出)...")
    keyboard.wait('esc')
    
    # 清理
    global running
    running = False
    if send_thread:
        send_thread.join(timeout=1.0)
    sock.close()
    print("\n程序已退出")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"主程序错误: {e}")
        import traceback
        traceback.print_exc()
        input("按Enter退出...")

