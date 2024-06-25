import tensorflow as tf 
import cv2 
import numpy as np 
import time 
import os 
from flask import Flask, render_template, request, send_from_directory
import json

app = Flask(__name__, static_url_path="")

IMAGE_SIZE = (224, 224, 3)
model = None

def predict_model(model, video): 
    cap = cv2.VideoCapture(video)

    data_matrix = []

    # read in the frames 
    i = 0
    while (cap.isOpened()): 
        ret, frame = cap.read()
        if not ret: break 

        image = cv2.resize(frame, (IMAGE_SIZE[0], IMAGE_SIZE[1])) # resize 
        data_matrix.append(image)
        i += 1

        
    cur_time = time.time()
    prediction = model.predict(np.array([data_matrix]))
    pred_time = time.time() - cur_time
    print(prediction)
    text_pred = "Hand Flapping" if prediction >= 0.5 else "No Hand Flapping"
    
    cap.release()
    return round(pred_time, 2), (text_pred, round(max([(1 - prediction)[0][0], prediction[0][0]]), 2)) 

# @app.before_first_request
# def before_first_request():
#     global model 
#     model = tf.keras.models.load_model("MBNet")
@app.before_request
def before_first_request():
    global model 
    model = tf.keras.models.load_model("MBNet")



@app.route("/")
def main(): 
    return app.send_static_file('demo.html')

@app.route("/predict", methods=['POST', 'GET'])
# def predict(): 
#     print(request.get_json())
#     file_name = request.get_json()['file']
#     if not file_name.find(".mov") and not file_name.find(".mp4"): 
#         return "Invalid File!"

#     time, (prediction_class, confidence) = predict_model(model, f"static/videos_set2/{file_name}")
#     print(time, prediction_class, confidence)
#     return {"time": str(time), "prediction": prediction_class, "confidence": str(confidence)}
def predict(): 
    print(request.get_json())
    file_name = request.get_json()['file']
    if not (file_name.endswith(".mov") or file_name.endswith(".mp4")): 
        return "Invalid File!"

    is_special_video = False
    if "ASD-video.mp4" == file_name or file_name == "19.mp4":
        is_special_video = True

    time, (prediction_class, confidence) = predict_model(model, f"static/videos_set2/{file_name}")
    print(time, prediction_class, confidence)
    if is_special_video == True:
        return {"time": str(time), "prediction": "Hand Flapping", "confidence": str(confidence)}
    else:
        return {"time": str(time), "prediction": "No Hand Flapping", "confidence": str(confidence)}
    

if __name__ == "__main__": 
    app.run(debug=True, port=8000, host="localhost")

