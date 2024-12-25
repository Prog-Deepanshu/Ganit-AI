# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
import google.generativeai as genai
import io
import base64
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)


genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')


cap = cv2.VideoCapture(0)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

prev_pos = None
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
output_text = ""

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is not None:
            cv2.line(canvas, tuple(map(int, prev_pos)), tuple(map(int, current_pos)), (255, 255 , 255), 10)
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up to clear the canvas
        canvas = np.zeros_like(canvas)
        prev_pos = None
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Four fingers up to send to AI
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

def generate_frames():
    global prev_pos, canvas, output_text
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        
        if img.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        
        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            new_text = sendToAI(model, canvas, fingers)
            if new_text:
                output_text = new_text

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        _, buffer = cv2.imencode('.jpg', image_combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_answer')
def get_answer():
    global output_text
    return jsonify({'answer': output_text})

if __name__ == "__main__":
    app.run(debug=True)