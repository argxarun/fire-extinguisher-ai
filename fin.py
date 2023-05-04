import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import serial,time

ArduinoSerial=serial.Serial('com6',9600,timeout=0.1)
labels=[]
cap=cv2.VideoCapture(0)
flag=0
while True:

    ret,img=cap.read()
    modelpath='detect.tflite'   
    min_conf=0.5
    
    labels=["fire"]
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
      
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
      
    float_input = (input_details[0]['dtype'] == np.float32)
      
    input_mean = 127.5
    input_std = 127.5
      
    image=img
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
      
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
      
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
      
    detections = []
      
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
      
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
    
            image=cv2.circle(image,(cx,cy),2,(0,0,255),2)

            string='X{0:d}Y{1:d}'.format((cx),(cy))
            print(string)
            ArduinoSerial.write(string.encode('utf-8'))
            
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
      
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
       
    cv2.rectangle(image,(640//2-30,480//2-30),(640//2+30,480//2+30),(255,255,255),3)        
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
     
          

    if cv2.waitKey(1) == ord("q"):
        break
      
cv2.destroyAllWindows()
cap.release()