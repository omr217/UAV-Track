# 🛸 YOLOv11-Based Object Detection & QR/Barcode Scanner

This project integrates real-time **object detection using a custom YOLOv11 model** with **QR/barcode recognition** on video frames. It provides region-of-interest (ROI) filtering, tracking logic for object selection, and average performance metrics like FPS and confidence.

---

## 🔍 Features

- ✅ Real-time object detection using **Ultralytics YOLOv11** (or v8 with custom weights)
- ✅ Tracks objects across frames based on proximity
- ✅ Defines and checks an ROI boundary box
- ✅ Decodes QR/barcodes using a user-defined method (`get_barcode_data`)
- ✅ Measures and displays:
  - Per-frame FPS
  - Average confidence of detected objects
- ✅ Works with videos, webcams, or other streams

---

## 🧠 Requirements

- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- Ultralytics YOLO (`pip install ultralytics`)
- A custom barcode utility module (`barcode.py`) with a `get_barcode_data` function

Install dependencies:
```bash
pip install opencv-python ultralytics torch
```

### 🧾 Project Structure
.
├── barcode.py              # Your barcode reading logic
├── weight/v11m_trained.pt  # Custom YOLOv11 model weights
├── UAV.mp4                 # Example video file
├── detection_script.py     # Main script (the one in this README)
└── README.md


#### 🧪 How It Works

**1. Initialization**
python
Kopyala
Düzenle
detect_model = Detection("UAV.mp4")
Accepts a video file or a camera index.

Loads YOLOv11 model and starts the capture stream.

**2. Object Detection**
YOLO model detects bounding boxes, classes, and confidence scores.

Draws only the box closest to the previously tracked box.

**3. ROI Check**
Defines a center rectangle as a region of interest.

Tracks if the selected object remains within the ROI.

**4. QR/Barcode Detection**
If the detected class is QR (e.g., class index 1) and it's in the ROI,
the frame is passed to get_barcode_data() for decoding.

python
```
def get_barcode_data(image):
    # Should return: (decoded_text, timestamp)
    return decoded_string, detection_time
```
**5. Performance Tracking**

FPS is calculated for each frame.

Confidence values are averaged over all detected objects.

##### 🧰 Additional Notes

YOLO weight path: weight/v11m_trained.pt – adjust as needed.

If you’re using Jetson devices, the script includes:
```
sudo systemctl restart nvargus-daemon
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
```
