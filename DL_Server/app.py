from http.client import responses
import json
from unittest import result
from flask import Flask, Response, jsonify, make_response, render_template, request
from flask_cors import CORS
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import requests

import cv2
import mediapipe as mp
from scipy import stats
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import ImageFont, ImageDraw, Image
import asyncio

app = Flask(__name__)
CORS(app)
sequence = []
sentence = []
predictions = []
# global results
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

###############
# Path for exported data, numpy arrays

DATA_PATH = os.path.join("DL_Server/MP_Data")

# Actions that we try to detect
actions = np.array(os.listdir(
    "DL_Server/MP_Data"))


###############
def createkorean():
    kor = {}
    kor['bruise'] = "멍이 들었습니다."
    kor['cough'] = "기침을 합니다."
    kor['default'] = "기본자세"
    kor['digestion'] = "소화"
    kor['heat'] = "열이 납니다."
    kor['hello'] = "안녕하세요"
    kor['injection'] = "주사"
    kor['muscle'] = "근육"
    kor['none'] = "없습니다"
    kor['sick'] = "아프다"
    kor['swell_up'] = "부었습니다."
    kor['thanks'] = "감사합니다."

    return kor


font = ImageFont.truetype('fonts/SCDream6.otf', 27)


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACE_CONNECTIONS)
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


model = load_model('20220606_90.h5')

colors = [(245, 117, 16) for _ in range(actions.size)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 70+num*20),
                      (int(prob*120), 90+num*20), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


@app.route('/video_feed')
def video_feed():
    status = '200 OK'
    return Response(result(), mimetype="multipart/x-mixed-replace; boundary=frame", status=status)


global res


@app.route('/', methods=['POST'])
async def video():
    print("in")

    info = request.files['file']
    # info.save('./videos/' + secure_filename(info.filename))
    path = './videos/' + secure_filename(info.filename)
    print(path)
    await asyncio.wait([getResult(path)])

    # print(info)

    status = '200 OK'
    return Response("", status=status)


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'GET':
        global res
        param = request.get_json()
        # param = json.loads(request.get_json(), encoding='utf-8')
        print("get in")
        print(res)
        # print(jsonify(param))
        return res
    if request.method == 'POST':
        # param = request.get_json()
        # print("IN")
        info = request.get_json()

        print(info)
        res = make_response(json.dumps(info, ensure_ascii=False))
        res.headers['Content-Type'] = 'application/json'
        return res


@app.route('/')
def index():
    return render_template('index.html')


def result():
    global sequence
    global sentence
    global predictions
    threshold = 0.88

    kor = createkorean()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    height, width = 1000, 1000
    outline = cv2.imread('y_512_2.png')
    size = 1000
    outline = cv2.resize(outline, (size, size))
    img_height, img_width, _ = outline.shape

    x, y = int(height/2 - img_height/2), int(width/2 - img_width/2)
    img2grey = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2grey, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    forground_with_mask = cv2.bitwise_and(outline, outline, mask=mask)

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read
            ret, frame = cap.read()

            # Make detections
            if ret:
                image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            # draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            # Vizualize
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 1:
                    sentence = sentence[-1:]

                # Vizualize
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

            if len(sentence) > 0:
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                draw.text(xy=(10, 10), text=kor[sentence[0]],
                          font=font, fill=(255, 255, 255))
                image = np.array(image)

            image = cv2.resize(image, (1000, 1000),
                               interpolation=cv2.INTER_LINEAR)

            imgArea = image[y:y+img_height, x:x+img_width]

            background_with_mask = cv2.bitwise_and(
                imgArea, imgArea, mask=mask_inv)

            final_img = cv2.add(forground_with_mask, background_with_mask)

            image[y:y+img_height, x:x+img_width] = final_img

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # show in WebPage
            re, buffer = cv2.imencode('.jpg', image)
            f = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n\r\n')
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="5500", debug=True,
    #         threaded=True)
    app.run(host="0.0.0.0", port="5500", debug=True,
            threaded=True, ssl_context=(cert.pem, key.pem))
