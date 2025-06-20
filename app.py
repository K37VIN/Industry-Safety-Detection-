import sys
import os
import shutil
from isd.pipeline.train_pipeline import TrainPipeline
from isd.exception import isdException
from isd.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()

# ✅ MANUAL MODEL PATH - use raw string or double backslashes to avoid issues
MANUAL_MODEL_PATH = r"E:\ISD\yolov7\runs\train\exp\weights\best.pt"

@app.route("/train")
def trainRoute():
    try:
        pipeline = TrainPipeline()
        # Run only ingestion & validation
        ingestion_artifact = pipeline.start_data_ingestion()
        pipeline.data_validation_artifact = pipeline.start_data_validation(ingestion_artifact)

        # ✅ After validation, push the manually placed model
        pipeline.push_manual_model(manual_model_path=MANUAL_MODEL_PATH)

        return jsonify({"message": "Data ingestion & validation completed. Manual model pushed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        # Use manual model path relative to yolov7 folder for detection
        weights_relative_path = os.path.relpath(MANUAL_MODEL_PATH, start="yolov7")
        source_image_path = os.path.join("data", clApp.filename)

        detect_cmd = (
            f'cd yolov7 && python detect.py --weights "{weights_relative_path}" '
            f'--source ../{source_image_path} --conf 0.25'
        )
        ret_code = os.system(detect_cmd)
        if ret_code != 0:
            return Response("Detection failed. Check server logs.", status=500)

        detect_runs_dir = os.path.join("yolov7", "runs", "detect")
        exp_folders = [f for f in os.listdir(detect_runs_dir) if f.startswith("exp")]
        if not exp_folders:
            return Response("No detection output folder found", status=500)

        # Get the latest exp folder by modification time
        latest_exp = sorted(exp_folders, key=lambda x: os.path.getmtime(os.path.join(detect_runs_dir, x)), reverse=True)[0]

        output_image_path = os.path.join(detect_runs_dir, latest_exp, clApp.filename)
        if not os.path.exists(output_image_path):
            return Response("Output image not found after detection", status=500)

        opencodedbase64 = encodeImageIntoBase64(output_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data", status=400)
    except KeyError:
        return Response("Key value error - incorrect key passed", status=400)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


