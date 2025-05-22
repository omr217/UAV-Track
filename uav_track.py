import os
import sys
import time
import math
import cv2
import torch
import subprocess
from pathlib import Path

# If you have a custom barcode or QR-code reading function:
from barcode import get_barcode_data

# Ultralytics YOLO (v8 or your "v11" version as you’ve named it)
from ultralytics import YOLO

class Detection:
    def __init__(self, source):
        """
        Initializes the Detection class with source and default parameters.
        """

        # Video input source (can be file path or camera index)
        self.source = source

        # Tracking-related variables
        self.old_target = [800, 800, 800, 800]  # Previous bounding box
        self.new_target = [800, 800, 800, 800]  # New bounding box for current frame
        self.min_dist = 1e9                     # Track closest bounding box
        self.in_area = False

        # ROI boundaries (example for a 1280x720 or 1600x900 resolution)
        # Adjust these as needed.
        self.kamera_width = 1280
        self.kamera_height = 720
        self.sol_sinir = int(0.25 * self.kamera_width)  # left boundary
        self.sag_sinir = int(0.75 * self.kamera_width)  # right boundary
        self.ust_sinir = int(0.1 * self.kamera_height)  # top boundary
        self.alt_sinir = int(0.9 * self.kamera_height)  # bottom boundary
        self.offset_ratio = 0.0

        # QR/Barcode detection
        self.qr_data = "0"
        self.bitis_zamani = "0"

        # FPS calculation
        self.prev_time = 0
        self.counter_frame = 0

        # Attempt to reset camera services (if you are using Jetson)
        # Remove if not needed.
        command_1 = 'sudo systemctl restart nvargus-daemon'
        command_2 = 'export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0'
        subprocess.call(command_1, shell=True)
        subprocess.call(command_2, shell=True)

        # Load the YOLO model
        # Replace "weight/v11m_trained.pt" with your actual model weights
        self.model = YOLO("weight/v11m_trained.pt")

        # Class names—adapt to your model’s classes
        self.class_names = ["UAV", "QR", "PLANE", "AIRCRAFT"]

    def check_intersection(self):
        """
        Checks if the current bounding box remains within certain ROI boundaries.
        Returns True if inside the defined area, otherwise False.
        """
        x1, y1, x2, y2 = self.old_target
        # Expand or shrink bounding box edges by offset_ratio if desired
        if x1 < (self.sol_sinir - self.offset_ratio * abs(x2 - x1)):
            return False
        if x2 > (self.sag_sinir + self.offset_ratio * abs(x2 - x1)):
            return False
        if y1 < (self.ust_sinir - self.offset_ratio * abs(y2 - y1)):
            return False
        if y2 > (self.alt_sinir + self.offset_ratio * abs(y2 - y1)):
            return False
        return True

    def scan_barcode(self, frame, x1, y1, x2, y2):
        """
        Crops the bounding box and tries to read a barcode or QR code.
        """
        cropped_img = frame[y1:y2, x1:x2]
        self.qr_data, self.bitis_zamani = get_barcode_data(cropped_img)
        if self.qr_data != "0":
            print(f"Decoded QR/Barcode: {self.qr_data}")

    def run_inference(self):
        """
        Main detection loop—reads frames from the source,
        runs YOLO inference, calculates FPS and confidence,
        performs bounding-box tracking, ROI checks, and optional barcode scanning.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: cannot open video {self.source}")
            return

        total_fps = 0.0
        frame_count = 0
        total_confidence = 0.0
        object_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                print("End of video or unable to read frame.")
                break

            # 1) Calculate FPS
            new_time = time.time()
            fps = 0
            if self.prev_time != 0:
                fps = 1 / (new_time - self.prev_time)
            self.prev_time = new_time

            total_fps += fps
            frame_count += 1

            # 2) Run YOLO inference
            results = self.model(frame, stream=True)

            # Reset min distance each frame for box selection
            self.min_dist = 1e9
            chosen_box = None

            # For average confidence in this frame
            frame_confidence_sum = 0.0
            frame_object_count = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # xyxy
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Confidence
                    conf = float(box.conf[0])
                    frame_confidence_sum += conf
                    frame_object_count += 1

                    # Class
                    cls = int(box.cls[0])
                    label = self.class_names[cls] if cls < len(self.class_names) else f"Unknown {cls}"

                    # Distance to old target center
                    old_cx = (self.old_target[0] + self.old_target[2]) / 2
                    old_cy = (self.old_target[1] + self.old_target[3]) / 2
                    new_cx = (x1 + x2) / 2
                    new_cy = (y1 + y2) / 2
                    dist = math.sqrt((old_cx - new_cx)**2 + (old_cy - new_cy)**2)

                    # Select the box closest to the previous target
                    if dist < self.min_dist:
                        self.min_dist = dist
                        chosen_box = (x1, y1, x2, y2, conf, cls)

            # 3) Draw and track only the chosen bounding box
            if chosen_box is not None:
                x1, y1, x2, y2, conf, cls = chosen_box
                self.old_target = [x1, y1, x2, y2]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

                # Put class label + confidence on frame
                label_text = f"{self.class_names[cls]} {conf:.2f}"
                cv2.putText(frame,
                            label_text,
                            (x1, max(25, y1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2)

                # Check ROI
                self.in_area = self.check_intersection()

                # Optional: If class == 1 (assumed QR) and in area -> scan
                # You can change '1' to whichever class index corresponds to QR.
                if cls == 1 and self.in_area:
                    self.scan_barcode(frame, x1, y1, x2, y2)

            # 4) Update total confidence counters
            if frame_object_count > 0:
                total_confidence += frame_confidence_sum
                object_count += frame_object_count

            # 5) Display the current FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the frame
            cv2.imshow("YOLOv11 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Final average FPS
        avg_fps = total_fps / frame_count if frame_count > 0 else 0
        # Final average confidence
        avg_conf = total_confidence / object_count if object_count > 0 else 0

        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Confidence: {avg_conf:.2f}")

if __name__ == "__main__":
    # Example usage: run on a local video file named "UAV.mp4" (or camera index "0" for webcam)
    detect_model = Detection("UAV.mp4")
    detect_model.run_inference()
