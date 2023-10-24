import cv2
import numpy as np
import picamera
import picamera.array
from flask import Flask, Response, render_template
import os
import time
import threading
from multiprocessing import Process
from firebase_admin  import credentials
from firebase_admin import storage
from uuid import uuid4
import subprocess
import firebase_admin
from multiprocessing import Process
from camara import generate_frames,camera_process,generate_frames1,start_camera_process,start_camera_process1
import atexit
app = Flask(__name__)




@app.route('/')
def index():
    start_camera_process()
    return render_template('pi2.html')

@app.route('/video_feed')
def video_feed():
    start_camera_process()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    atexit.register(generate_frames1)
    app.run(host='0.0.0.0', port=8000, debug=True)