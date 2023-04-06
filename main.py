import os
import dlib
import cv2
import serial
import numpy as np
from imutils import face_utils
from urllib.request import urlretrieve
from bz2 import decompress

# config (depends on the camera)
h_FOV = 60
v_FOV = 40

cam_width = 640
cam_height = 480

center_x = round(cam_width / 2)
center_y = round(cam_height / 2)

def check_model():
    model_url = f"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
    model_name = "shape_predictor_68_face_landmarks_GTX.dat"
    if not os.path.exists(model_name):
        urlretrieve(model_url, model_name + '.bz2')
        with open(model_name, 'wb') as decompressed_file, open(model_name + '.bz2', 'rb') as commpressed_file:
            data = decompress(commpressed_file.read())
            decompressed_file.write(data)
        os.remove(model_name + '.bz2')
    
    return model_name

def main():
    model_name = check_model()
    
    ser = serial.Serial("/dev/ttyTHS1", 115200)
    predictor = dlib.shape_predictor(model_name)
    detector = dlib.get_frontal_face_detector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        try:
            _, frame = cap.read()
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (cam_width, cam_height))
            
            rects = detector(resized, 0)
            for _, rect in enumerate(rects):
                shape = predictor(resized, rect)
                shape = face_utils.shape_to_np(shape)
                
                mouth = shape[49:61]
                x_min, y_min = mouth.min(axis=0)
                x_max, y_max = mouth.max(axis=0)
                width, height = (x_max - x_min), (y_max - y_min)
                x_mid, y_mid = round((x_max + x_min) / 2), round((y_max + y_min) / 2)
                h_angle , v_angle = ((x_mid - center_x) / center_x) * (h_FOV / 2)
                
                ser.write("{0} {1} {2} {3}\r\n".format(width, height, h_angle, v_angle).encode())
        except KeyboardInterrupt:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()