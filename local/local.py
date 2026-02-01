import cv2
import numpy as np
import json
import os
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CẤU HÌNH & DATABASE
# Windows: "C:/Windows/Fonts/arial.ttf"
# macOS: "/Library/Fonts/Arial.ttf"
# Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# ==========================================
DB_FILE = 'local/data/face_database.json'
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 
WINDOW_NAME = "Face Recognition - Local Mode"

face_database = {}

def load_db():
    global face_database
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            face_database = json.load(f)
        print(f"✅ Đã tải {len(face_database)} khuôn mặt từ database.")

def save_db():
    os.makedirs('local/data', exist_ok=True)
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(face_database, f, ensure_ascii=False, indent=4)
    print("💾 Đã lưu thay đổi vào database.")

def draw_vn_text(img, text, position, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 24)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH & CAMERA
# ==========================================
load_db()

# Tận dụng GPU cho model Buffalo_L
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(WINDOW_NAME, 640, 480) # Tùy chỉnh kích thước cửa sổ 
is_fullscreen = False

print("\n--- PHÍM TẮT ĐIỀU KHIỂN ---")
print("'f': Toàn màn hình \n'r': Đăng ký người mới \n'd': Xóa người trong database \n'q': Thoát")

# ==========================================
# 3. VÒNG LẶP XỬ LÝ
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    faces = app.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        
        name = "Người lạ"
        max_sim = -1
        
        # Tính toán độ tương đồng Cosine
        for db_name, db_emb in face_database.items():
            sim = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
            if sim > max_sim:
                max_sim = sim
                if sim > 0.45:
                    name = db_name
        
        color_bgr = (0, 255, 0) if name != "Người lạ" else (0, 0, 255)
        color_rgb = (0, 255, 0) if name != "Người lạ" else (255, 0, 0)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)
        
        label = f"{name} ({max_sim:.2f})"
        frame = draw_vn_text(frame, label, (bbox[0], bbox[1] - 35), color_rgb)

    cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # --- LOGIC PHÍM BẤM ---
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            
    elif key == ord('r'):
        if len(faces) == 1:
            new_name = input("✍️ Nhập tên mới: ")
            if new_name:
                face_database[new_name] = faces[0].embedding.tolist()
                save_db()
                print(f"✅ Đã đăng ký: {new_name}")
        else:
            print("⚠️ Lỗi: Chỉ đăng ký khi có duy nhất 1 khuôn mặt trong khung hình.")

    # ĐÂY LÀ PHẦN XÓA MỚI BỔ SUNG
    elif key == ord('d'):
        name_to_del = input("🗑️ Nhập tên người muốn xóa: ")
        if name_to_del in face_database:
            confirm = input(f"❓ Bạn có chắc chắn muốn xóa '{name_to_del}'? (y/n): ")
            if confirm.lower() == 'y':
                del face_database[name_to_del]
                save_db()
                print(f"❌ Đã xóa thành công: {name_to_del}")
        else:
            print(f"❌ Không tìm thấy tên '{name_to_del}' trong database.")
                
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()