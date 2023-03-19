import os
import dlib
import cv2
import numpy as np
from imutils import face_utils
from urllib.request import urlretrieve
from bz2 import decompress

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
    predictor = dlib.shape_predictor(model_name)
    detector = dlib.get_frontal_face_detector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        for _, rect in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            mouth = shape[49:61]
            x_min, y_min = mouth.min(axis=0)
            x_max, y_max = mouth.max(axis=0)
            x_mid, y_mid = round((x_max + x_min) / 2), round((y_max + y_min) / 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
            cv2.circle(frame, (x_mid, y_mid), 4, (255, 0, 0), -1)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()