## Industry-Safety-Detection 👷‍♂️🏗️🚧

An AI-powered web application that leverages the YOLOv7 deep learning model and a Flask-based REST API to detect industrial safety compliance such as PPE usage, unauthorized access, and hazardous conditions in real-time. The system supports automated data ingestion and validation pipelines while allowing manual integration of pretrained models, making it flexible for retraining or deployment without reinitializing the training process. It accepts images via a base64-encoded API payload, processes them using the YOLOv7 model, and returns annotated output images, making it suitable for industrial monitoring dashboards or safety automation systems. The application is designed for cloud deployment (e.g., Render) and provides endpoints for both prediction and pipeline management.

## Tech Stack:

- Flask
- Yolo V7
- OpenCV
- Render

## Clone the repository:

```
git clone https://github.com/K37VIN/Industry-Safety-Detection-.git
```

## Install the requirements:

```
pip install -r requirements.txt
```

## Run the application:

```
python app.py
```

## Demo of the app:
https://drive.google.com/drive/folders/1qNwsEOYFKILZUNEji_exOoYFUQ9Z5Jot?usp=sharing
