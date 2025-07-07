# video_classification.py
import os, cv2, time, uuid, threading, queue, numpy as np
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D
from queue import Queue
import camera_receiver

# ---------- 去眩光增强函数  (### <<< 新增 >>>)
def _bright_lut(th=220, k=0.45):
    lut = np.arange(256, dtype=np.uint8)
    lut[th:] = (th + (lut[th:] - th) * k).astype(np.uint8)
    return lut
_LUT_BASE = _bright_lut(220, 0.45)

def _apply_large_area(v, area_th=2000):
    mask = (v > 230).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    lut_big = _bright_lut(200, 0.30)
    v_big   = cv2.LUT(v, lut_big)
    v_out   = v.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > area_th:
            v_out[lbl == i] = v_big[lbl == i]
    return v_out

def deglare_enhance(bgr):
    """温和去反光 + 轻锐化"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.LUT(v, _LUT_BASE)
    v = _apply_large_area(v)
    res = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    blur = cv2.GaussianBlur(res, (5, 5), 0)
    return cv2.addWeighted(res, 1.15, blur, -0.15, 0)
# ---------- 去反光函数结束 ----------


SAVE_DIR = "./captured_cats"
YOLO_MODEL_PATH = "./yolov8n.pt"
CLASSIFY_MODEL_PATH = "./model/mix_dataset_model.h5"
CLASSES = ['Pallas cats','Persian cats','Ragdolls','Singapura cats','Sphynx cats']
os.makedirs(SAVE_DIR, exist_ok=True)

# 模型实例
detector = None
classifier = None

# 最后预测结果
last_prediction = {"label": "", "conf": 0.0}

class DepthwiseConv2DCompat(DepthwiseConv2D):
    def __init__(self,*a,groups=1,**k): super().__init__(*a,**k)

def initialize_models():
    """初始化模型"""
    global detector, classifier
    print("初始化YOLO模型...")
    detector = YOLO(YOLO_MODEL_PATH)
    
    print("初始化EfficientNetB3分类模型...")
    classifier = tf.keras.models.load_model(
        CLASSIFY_MODEL_PATH,
        compile=False, 
        custom_objects={'DepthwiseConv2D': DepthwiseConv2DCompat}
    )
    print("模型初始化完成")

def pad_resize(img, size=300):
    """调整图像大小并填充"""
    h,w = img.shape[:2]; s = size/max(h,w)
    nh,nw = int(h*s),int(w*s)
    canvas = np.full((size,size,3),255,np.uint8)
    resized = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    y0,x0 = (size-nh)//2, (size-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def classify_cat(img):
    """分类猫的品种"""
    arr = preprocess_input(cv2.resize(img,(300,300)).astype("float32"))
    p = classifier.predict(np.expand_dims(arr,0),verbose=0)[0]
    idx = int(np.argmax(p)); 
    return CLASSES[idx], float(p[idx])

def process_frame(frame, conf_th=0.25, cls_disp_th=0.70):
    """处理单帧图像"""
    global last_prediction
    start = time.time() 
    
    # 使用YOLO检测猫
    frame = deglare_enhance(frame)

    res = detector.predict(frame, conf=conf_th, verbose=False)[0]
    best_box, best_conf = None, 0.0
    
    for b in res.boxes:
        if int(b.cls) == 15 and b.conf > best_conf:  # 15是猫的类别ID
            best_box, best_conf = b, float(b.conf)

    if best_box is not None:
        x1,y1,x2,y2 = map(int, best_box.xyxy[0])

        # 绘制边界框和置信度
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"Cat {best_conf:.2f}",(x1,max(0,y1-25)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # 如果置信度足够高，显示品种
        if last_prediction["conf"] >= cls_disp_th:
            txt = f"{last_prediction['label']} ({last_prediction['conf']:.2f})"
            cv2.putText(frame, txt, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        # 保存图像并进行分类
        def worker(img, box):
            xi,yi,xa,ya = box
            crop = img[yi:ya, xi:xa]
            padded = pad_resize(crop)
            fname = f"cat_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR,fname), padded)
            lbl, cf = classify_cat(padded)
            last_prediction["label"] = lbl
            last_prediction["conf"] = cf
            print(f"✔ Saved {fname} | {lbl} ({cf:.2f})")
        
        threading.Thread(target=worker,args=(frame.copy(),(x1,y1,x2,y2)),daemon=True).start()

        latency = (time.time() - start) * 1000
        cv2.putText(frame, f"{latency:.0f} ms", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

def get_last_prediction():
    """获取最后的预测结果"""
    if last_prediction["conf"] >= 0.7:
        return last_prediction["label"], last_prediction["conf"]
    else:
        return "", 0.0  # 返回空字符串和0置信度

def generate_processed_frames():
    """生成处理后的视频帧"""
    frame_queue = camera_receiver.get_frame_queue()
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = process_frame(frame)
            
            # 转换为JPEG格式
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # 如果没有新帧，等待一段时间
            time.sleep(0.05)