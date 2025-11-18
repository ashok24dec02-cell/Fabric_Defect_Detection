# Fabric Defect Detection App (React Native + Django + YOLOv8)

This project is a full-stack mobile application for detecting fabric defects using a trained YOLOv8 model.  
Users upload a fabric image through the mobile app, the Django backend processes it, and the API returns an annotated image showing detected defects.

------------------------------------------------------------
1. FEATURES
------------------------------------------------------------

Frontend (React Native)
- Upload fabric image from gallery or camera
- Sends image to backend for detection
- Displays annotated output image
- Simple and user-friendly UI

Backend (Django REST API)
- YOLOv8 inference for fabric defects
- Accepts image via POST request
- Returns JSON with defect data and annotated image URL
- Saves uploaded and output images

YOLOv8 Model
- Trained on custom fabric defect dataset
- Detects defects such as:
  hole, stain, scratch, yarn_defect, color_variation

------------------------------------------------------------
2. PROJECT STRUCTURE
------------------------------------------------------------

fabric-defect-detection/
│
├── backend/          (Django + YOLOv8)
│   ├── detect/
│   ├── model/best.pt
│   ├── manage.py
│   └── requirements.txt
│
└── mobile/           (React Native)
    ├── App.js
    ├── screens/
    ├── components/
    └── package.json

------------------------------------------------------------
3. BACKEND SETUP (DJANGO + YOLOv8)
------------------------------------------------------------

Install dependencies:
    cd backend
    pip install -r requirements.txt

Start server:
    python manage.py runserver

Backend API endpoint:
    POST http://localhost:8000/api/detect/

Expected API response:
{
  "status": "success",
  "defects": [
    {"label": "hole", "confidence": 0.91},
    {"label": "stain", "confidence": 0.83}
  ],
  "annotated_image_url": "http://localhost:8000/media/output/annotated.jpg"
}

------------------------------------------------------------
4. FRONTEND SETUP (REACT NATIVE)
------------------------------------------------------------

Install dependencies:
    cd mobile
    npm install

Start development server:
    npx expo start

Update backend URL in code:
    const API_URL = "http://<your-ip>:8000/api/detect/";

------------------------------------------------------------
5. YOLOv8 MODEL DETAILS
------------------------------------------------------------

Training command example:
    model = YOLO("yolov8m.pt")
    model.train(
        data="fabric.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        device=0
    )

Trained model is stored at:
    backend/model/best.pt

------------------------------------------------------------
6. API USAGE
------------------------------------------------------------

POST /api/detect/
Form-data:
  Key: image
  Type: File
  Description: Fabric image to check for defects

The API returns:
- Detected defect names
- Confidence scores
- Annotated image URL
- Bounding box details

------------------------------------------------------------
7. REQUIREMENTS
------------------------------------------------------------

Backend Requirements:
- Python 3.x
- Django
- Django REST Framework
- Ultralytics (YOLOv8)
- OpenCV
- Pillow

Frontend Requirements:
- React Native

------------------------------------------------------------
8. DEPLOYMENT
------------------------------------------------------------

Backend deployment options:
- Render
- Railway
- AWS EC2
- DigitalOcean

Mobile deployment:
- Expo Build
- Google Play Store
- Apple App Store

------------------------------------------------------------
9. LICENSE
------------------------------------------------------------

This project is released under the MIT License.
