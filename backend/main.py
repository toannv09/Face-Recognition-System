from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import io
import cv2
import base64
from typing import List, Dict
import json
import os

app = FastAPI(title="Face Recognition API", version="1.0.0")

# CORS - cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phát hiện môi trường
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Khởi tạo InsightFace model (chạy trên CPU)
print("Loading InsightFace models...")
# Trên production dùng model nhỏ hơn để tiết kiệm RAM
model_name = 'buffalo_s' if IS_PRODUCTION else 'buffalo_l'
det_size = None  # Avoid forcing a square detection size that can distort AR for non-square images

face_app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
# Prepare without a fixed det_size so the detector can work with the image's native aspect ratio
if det_size is None:
    face_app.prepare(ctx_id=0)
    print(f"Models loaded successfully! (Model: {model_name}, Detection size: auto)")
else:
    face_app.prepare(ctx_id=0, det_size=det_size)
    print(f"Models loaded successfully! (Model: {model_name}, Detection size: {det_size})")

# Database đơn giản lưu embeddings 
face_database: Dict[str, List[float]] = {}

# Load database từ file nếu có (cho production). Sử dụng thư mục `data/` để dễ mount volume
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, 'face_database.json')

# Load database if exists
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            face_database = loaded
            print(f"Loaded {len(face_database)} faces from database file: {DB_FILE}")
        else:
            print(f"Database file has unexpected format, ignoring: {DB_FILE}")
    except Exception as e:
        print(f"Could not load database: {e}")

# Thread-safe atomic save
from threading import Lock
_db_lock = Lock()

def save_database():
    """Lưu database vào file một cách an toàn (atomic, thread-safe)"""
    try:
        with _db_lock:
            temp_file = DB_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(face_database, f)
            os.replace(temp_file, DB_FILE)
            print(f"Saved database to {DB_FILE} ({len(face_database)} entries)")
    except Exception as e:
        print(f"Could not save database: {e}")


@app.get("/")
async def serve_home():
    """Serve trang chủ HTML"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "message": "Face Recognition API",
        "endpoints": {
            "/detect": "Phát hiện khuôn mặt trong ảnh",
            "/register": "Đăng ký khuôn mặt mới",
            "/recognize": "Nhận dạng khuôn mặt",
            "/recognize-frame": "Nhận dạng khuôn mặt realtime từ webcam",
            "/database": "Xem danh sách đã đăng ký"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Phát hiện khuôn mặt trong ảnh"""
    try:
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_array)
        
        if len(faces) == 0:
            return JSONResponse({
                "success": False,
                "message": "Không tìm thấy khuôn mặt nào",
                "count": 0
            })
        
        # Trả về thông tin các khuôn mặt
        results = []
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "face_id": idx,
                "bbox": bbox,  # [x1, y1, x2, y2]
                "confidence": float(face.det_score),
                "landmarks": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None
            })
        
        return JSONResponse({
            "success": True,
            "count": len(faces),
            "faces": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    """Đăng ký khuôn mặt mới vào database"""
    try:
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect và extract embedding
        faces = face_app.get(img_array)
        
        if len(faces) == 0:
            return JSONResponse({
                "success": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh"
            })
        
        if len(faces) > 1:
            return JSONResponse({
                "success": False,
                "message": f"Tìm thấy {len(faces)} khuôn mặt. Vui lòng upload ảnh chỉ có 1 khuôn mặt"
            })
        
        # Lưu embedding vào database
        embedding = faces[0].embedding.tolist()
        face_database[name] = embedding
        
        # Lưu vào file (cho production persistence)
        save_database()
        
        return JSONResponse({
            "success": True,
            "message": f"Đã đăng ký khuôn mặt cho '{name}'",
            "total_registered": len(face_database)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Nhận dạng khuôn mặt từ database"""
    try:
        if len(face_database) == 0:
            return JSONResponse({
                "success": False,
                "message": "Database trống. Vui lòng đăng ký khuôn mặt trước."
            })
        
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        image_size = {"width": image.width, "height": image.height}
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_array)
        
        if len(faces) == 0:
            return JSONResponse({
                "success": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh"
            })
        
        # Nhận dạng từng khuôn mặt
        results = []
        for idx, face in enumerate(faces):
            embedding = face.embedding
            
            # So sánh với database
            best_match = None
            best_similarity = -1
            
            for name, stored_embedding in face_database.items():
                # Tính cosine similarity
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Threshold: 0.4-0.6 tùy độ nghiêm ngặt
            threshold = 0.5
            if best_similarity > threshold:
                results.append({
                    "face_id": idx,
                    "name": best_match,
                    "similarity": float(best_similarity),
                    "bbox": face.bbox.astype(int).tolist()
                })
            else:
                results.append({
                    "face_id": idx,
                    "name": "Unknown",
                    "similarity": float(best_similarity),
                    "bbox": face.bbox.astype(int).tolist()
                })
        
        return JSONResponse({
            "success": True,
            "count": len(faces),
            "results": results,
            "image_size": image_size
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/recognize-frame")
async def recognize_frame(frame_data: dict):
    """Nhận dạng khuôn mặt realtime từ webcam frame (base64) - Optimized"""
    try:
        if len(face_database) == 0:
            return JSONResponse({
                "success": False,
                "message": "Database trống. Vui lòng đăng ký khuôn mặt trước.",
                "results": []
            })
        
        # Decode base64 image
        image_data = frame_data.get('image', '')
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image để xử lý nhanh hơn (giảm từ full resolution xuống)
        max_width = 640
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.LANCZOS)
        
        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces với det_size nhỏ hơn để nhanh hơn
        faces = face_app.get(img_array)
        
        if len(faces) == 0:
            return JSONResponse({
                "success": True,
                "count": 0,
                "results": [],
                "image_size": {"width": image.width, "height": image.height}
            })
        
        # Nhận dạng từng khuôn mặt
        results = []
        for idx, face in enumerate(faces):
            embedding = face.embedding
            
            # So sánh với database
            best_match = None
            best_similarity = -1
            
            for name, stored_embedding in face_database.items():
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Threshold có thể điều chỉnh
            threshold = 0.45  # Giảm một chút để nhận dạng dễ hơn
            
            # Làm tròn bbox để chính xác hơn
            bbox = face.bbox.astype(int).tolist()
            
            results.append({
                "face_id": idx,
                "name": best_match if best_similarity > threshold else "Unknown",
                "similarity": float(best_similarity),
                "bbox": bbox,  # [x1, y1, x2, y2]
                "confidence": float(face.det_score)
            })
        
        return JSONResponse({
            "success": True,
            "count": len(faces),
            "results": results,
            "image_size": {"width": image.width, "height": image.height}
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error: {str(e)}",
            "results": []
        })


@app.get("/database")
async def get_database():
    """Xem danh sách người đã đăng ký"""
    return JSONResponse({
        "success": True,
        "count": len(face_database),
        "names": list(face_database.keys())
    })


@app.delete("/database/{name}")
async def delete_from_database(name: str):
    """Xóa người khỏi database"""
    if name in face_database:
        del face_database[name]
        save_database()  # Lưu sau khi xóa
        return JSONResponse({
            "success": True,
            "message": f"Đã xóa '{name}' khỏi database"
        })
    else:
        return JSONResponse({
            "success": False,
            "message": f"Không tìm thấy '{name}' trong database"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)