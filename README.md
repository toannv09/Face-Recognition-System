# Face Recognition System 🔍

Ứng dụng nhận dạng khuôn mặt đơn giản sử dụng InsightFace (ONNX) để phát hiện và so khớp khuôn mặt. Dự án có hai chế độ hoạt động chính:

- **Backend (API)**: REST API viết bằng FastAPI để upload ảnh, phát hiện, đăng ký và nhận dạng khuôn mặt.
- **Local (Realtime OpenCV)**: Script Python dùng OpenCV để hiển thị luồng webcam và thực hiện nhận dạng theo thời gian thực, điều khiển bằng bàn phím.

**Live demo:** 

Trải nghiệm trực tiếp trên HuggingFace Spaces: https://huggingface.co/spaces/Lippovn04/Face-Recognition-System

> Lưu ý: bản demo chạy trên tài nguyên CPU chia sẻ (HuggingFace Spaces) và **không dành cho inference thời gian thực**.

---

## 🧠 Triết lý thiết kế

Dự án được thiết kế với nhiều chế độ thực thi để minh hoạ các đánh đổi thực tế giữa hiệu năng, tiện lợi triển khai và khả năng tiếp cận:

- Hiệu năng realtime chỉ đạt được khi chạy local với quyền truy cập trực tiếp vào phần cứng (webcam, GPU).
- Các triển khai dạng API hoặc đám mây ưu tiên tính di động và khả năng tái tạo hơn là tối ưu cho độ trễ thấp.

---

## ⚙️ Tính năng chính

- Phát hiện khuôn mặt trong ảnh (endpoint `/detect`)
- Đăng ký khuôn mặt mới (`/register`)
- Nhận dạng khuôn mặt so với database (`/recognize`, `/recognize-frame`)
- Giao diện local realtime dùng webcam (`local/local.py`) để đăng ký và nhận dạng
- Lưu trữ database dưới dạng JSON (dễ mount/backup)

---

## 📁 Cấu trúc dự án

```
./
├─ backend/             # FastAPI server
│  ├─ main.py
│  ├─ requirements.txt
│  └─ data/face_database.json
├─ local/               # Script local realtime (OpenCV)
│  ├─ local.py
│  └─ requirements.txt
├─ frontend/            # static frontend (tùy chọn)
└─ docker-compose.yml
```

---

## 🚀 Cách chạy

### Chạy nhanh bằng Docker (khuyến nghị để tránh cài đặt thủ công)

1. Cài Docker & docker-compose
2. Từ thư mục gốc dự án chạy:

```bash
docker-compose up --build
```

- Backend sẽ chạy ở `http://localhost:8000`
- Frontend (nếu dùng) phục vụ static trên `http://localhost:3001`

### Dùng image từ Docker Hub (pull)

Bạn có thể kéo image trực tiếp từ Docker Hub (đã được push sẵn):

```bash
# Kéo image
docker pull toannguyenuit/face-recognition:latest

# Chạy container (ví dụ):
docker run --name face-recognition -p 8000:8000 \
  -v $(pwd)/backend/data:/app/data \
  -v $(pwd)/backend/models:/root/.insightface/models \
  -e ENVIRONMENT=production \
  toannguyenuit/face-recognition:latest
```

- Trên Windows PowerShell, thay `$(pwd)` bằng `${PWD}` hoặc đường dẫn tuyệt đối.
- Nếu image hỗ trợ GPU và máy chủ của bạn có NVIDIA Container Toolkit, thêm `--gpus all` để kích hoạt GPU.

Bạn cũng có thể dùng image trong `docker-compose.yml` bằng cách thay `build:` bằng `image:` (ví dụ `toannguyenuit/face-recognition:latest`).

Kiểm tra container và logs:

```bash
docker ps
docker logs -f face-recognition
```

---

### Chạy local (Python)

1. Tạo virtualenv và kích hoạt
2. Cài dependencies cho backend và/hoặc local:

```bash
pip install -r backend/requirements.txt
pip install -r local/requirements.txt
```

3a. Chạy backend (trong thư mục `backend`):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# hoặc: python main.py
```

3b. Chạy GUI local (yêu cầu webcam):

```bash
python local/local.py
```

> Gợi ý: nếu muốn dùng GPU (CUDA) để tăng tốc inference, cài `onnxruntime-gpu` thay vì `onnxruntime`.

---

## 🔌 API chính (tóm tắt)

API được thiết kế cho inference dựa trên ảnh và mục đích thử nghiệm, **không tối ưu cho streaming video thời gian thực với độ trễ thấp**.

- `GET /` : trang chủ hoặc file `index.html` nếu có
- `POST /detect` : upload file ảnh để phát hiện khuôn mặt
- `POST /register` : đăng ký tên + ảnh (lưu embedding vào database)
- `POST /recognize` : nhận dạng 1 ảnh so với database
- `POST /recognize-frame` : nhận dạng realtime từ frame base64 (dành cho frontend)
- `GET /database` : liệt kê tên đã đăng ký
- `DELETE /database/{name}` : xóa một bản ghi

Ví dụ curl để detect:

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@/path/to/img.jpg"
```

---

## 🗂️ Database & Models

- Database (JSON): `backend/data/face_database.json` (server) và `local/data/face_database.json` (local)
- Models InsightFace được lưu tại `backend/models` (Docker volume mapping tới `/root/.insightface/models`)
- Mặc định backend dùng model `buffalo_l` khi phát triển, `buffalo_s` khi đặt biến môi trường `ENVIRONMENT=production` để tiết kiệm RAM

---

## ⚠️ Lưu ý & Khắc phục sự cố

- Nếu GUI không hiển thị chữ tiếng Việt đúng, kiểm tra `FONT_PATH` trong `local/local.py` (Windows mặc định `C:/Windows/Fonts/arial.ttf`)
- Nếu webcam không được phát hiện, kiểm tra camera index hoặc quyền truy cập
- Nếu gặp lỗi tương thích ONNX/onnxruntime trên hệ của bạn, thử cài `onnxruntime-gpu` hoặc điều chỉnh phiên bản phù hợp với Python

---

