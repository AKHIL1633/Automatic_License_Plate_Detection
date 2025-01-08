import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
import os
from pathlib import Path
from datetime import datetime
from streamlit_webrtc import webrtc_streamer
import av
import sqlite3
import json  # Import json for serialization and deserialization

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "backend_data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_type TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                detection_results TEXT
            )
        """)
        conn.commit()

# Call this function at the start of your application
init_db()

def insert_image_record(image_type, image_path, detection_results=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    absolute_image_path = str(image_path.resolve())  # Absolute path of image
    
    # Convert detection_results to a JSON string if it's not None
    if detection_results is not None:
        detection_results = json.dumps(detection_results)
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO images (image_type, image_path, timestamp, detection_results)
            VALUES (?, ?, ?, ?)
        """, (image_type, absolute_image_path, timestamp, detection_results))
        conn.commit()
    print(f"Inserted image record: {image_type}, {absolute_image_path}, {timestamp}")

def fetch_image_records():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images")
        return cursor.fetchall()

def display_images_from_db():
    # Fetch image records from the database and display images
    records = fetch_image_records()
    for record in records:
        image_path = record[2]  # image_path column is the 3rd element in the row
        detection_results = record[4]  # detection_results is the 5th element in the row
        
        # If detection_results is not None, parse it from JSON
        if detection_results:
            detection_results = json.loads(detection_results)
        
        print(f"Fetched image path: {image_path}")  # Debug: print the image path to verify it's correct
        if os.path.exists(image_path):
            st.image(image_path, caption="Stored Image", use_column_width=True)
        else:
            st.error(f"Image not found: {image_path}")  # Error if the image path doesn't exist
        
        if detection_results:
            st.json(detection_results)  # Display the detection results in JSON format

# Define base directories using pathlib
BASE_DIR = Path(__file__).resolve().parent

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
vehicles = [2]  # Vehicle classes for COCO model (class ID 2 usually refers to cars)

# Load YOLO models
LICENSE_MODEL_DETECTION_DIR = BASE_DIR / "models" / "license_plate_detector.pt"
COCO_MODEL_DIR = BASE_DIR / "models" / "yolov8n.pt"
coco_model = YOLO(str(COCO_MODEL_DIR))
license_plate_detector = YOLO(str(LICENSE_MODEL_DETECTION_DIR))

# Date-based directory setup
def get_datewise_folder(base_path):
    today_date = datetime.now().strftime("%Y-%m-%d")
    datewise_folder = base_path / today_date
    datewise_folder.mkdir(parents=True, exist_ok=True)
    return datewise_folder

# Updated folder path for cropped images
FOLDER_PATH = get_datewise_folder(BASE_DIR / "licenses_plates_imgs_detected")

# Date-wise folders for images and CSV files
CSV_DETECTION_PATH = get_datewise_folder(BASE_DIR / "csv_detections") / "detection_results.csv"

def save_detection_to_csv(detection):
    """
    Save detection results to a date-wise CSV. If the CSV file does not exist, create it.
    """
    df = pd.DataFrame([detection])
    # Save the cropped image with an absolute path
    cropped_image_path = Path(detection['cropped_image']).resolve()  # Use absolute path
    detection['cropped_image'] = str(cropped_image_path)
    
    if CSV_DETECTION_PATH.exists():
        df.to_csv(CSV_DETECTION_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_DETECTION_PATH, index=False)

def read_license_plate(license_plate_crop):
    """
    Use EasyOCR to extract text from the cropped license plate image.
    """
    detections = reader.readtext(license_plate_crop)
    plate = []

    for result in detections:
        bbox, text, score = result
        text = text.upper()
        plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), score
    else:
        return None, 0

def model_prediction(img):
    """
    Run YOLO models to detect vehicles and license plates. Save results and images.
    """
    license_numbers = 0
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Object Detection (Vehicle detection)
    object_detections = coco_model(img)[0]

    # License Plate Detection
    license_detections = license_plate_detector(img)[0]

    for license_plate in license_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)

        # Generate a unique filename for cropped images
        img_name = f'{uuid.uuid4()}.jpg'
        cropped_image_path = FOLDER_PATH / img_name
        cv2.imwrite(str(cropped_image_path), license_plate_crop)

        # Prepare detection details for CSV
        detection_data = {
            'license_number': license_plate_text,
            'text_score': license_plate_text_score,
            'bbox': [x1, y1, x2, y2],
            'detection_score': score,
            'cropped_image': str(cropped_image_path)
        }

        licenses_texts.append(license_plate_text)
        save_detection_to_csv(detection_data)

        # Insert the image record into the DB
        insert_image_record("detected", cropped_image_path, detection_data)

    img_with_boxes = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [img_with_boxes, licenses_texts]

# Streamlit UI setup
header = st.container()
body = st.container()

with header:
    st.title("License Plate Detection System")

with body:
    st.subheader("Select an option:")

    option = st.radio("Detection Mode", ("Upload an Image", "Take a Photo", "Live Detection"))

    if option == "Upload an Image":
        img = st.file_uploader("Upload a Car Image", type=["png", "jpg", "jpeg"])
    elif option == "Take a Photo":
        img = st.camera_input("Capture a Photo")
    elif option == "Live Detection":
        webrtc_streamer(key="live", video_processor_factory=lambda: VideoProcessor())  # type: ignore
        img = None

    if img is not None:
        image = np.array(Image.open(img))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Apply Detection"):
            results = model_prediction(image)
            st.image(results[0], caption='Detection Result', use_column_width=True)

            # Display stored CSV results
            if CSV_DETECTION_PATH.exists():
                df = pd.read_csv(CSV_DETECTION_PATH)
                st.dataframe(df)

            # Display images stored in DB from the database
            display_images_from_db()  # Display images stored in DB
