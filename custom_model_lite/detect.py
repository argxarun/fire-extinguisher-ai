# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import serial,time
import matplotlib
import matplotlib.pyplot as plt

ArduinoSerial=serial.Serial('com9',9600,timeout=0.1)
labels=[]
### Define function for inferencing with TFLite model and displaying results
cap=cv2.VideoCapture(0)
while True:

    ret,img=cap.read()
    modelpath='detect.tflite'  
    #lblpath='D:\DL\fire\custom_model_lite\labelmap.txt' 
    min_conf=0.5
    # Grab filenames of all images in test folder
    #images = glob.glob(imgpath + '/.jpg') + glob.glob(imgpath + '/.JPG') + glob.glob(imgpath + '/.png') + glob.glob(imgpath + '/.bmp')
      
    # Load the label map into memory
    # with open(lblpath, 'r') as f:
    #     labels = [line.strip() for line in f.readlines()]
    
    labels=["fire"]
    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
      
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
      
    float_input = (input_details[0]['dtype'] == np.float32)
      
    input_mean = 127.5
    input_std = 127.5
      
    # Randomly select test images
    #images_to_test = random.sample(images, num_test_images)
  

    # Load image and resize to expected shape [1xHxWx3]
    #image = cv2.imread(image_path)
    image=img
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
      
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
      
    print(input_details[0])    
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
      
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
      
    detections = []
      
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
      
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            x = xmin / width
            y = ymin / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            #plot the center of the face
            
            #plot the roi
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            #plot the squared region in the center of the screen
            x1=640//2-30
            y1=480//2-30
            x2=640//2+30
            y2=480//2+30
            #cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),3)
            # function call
            # if FindPoint(x1, y1, x2,y2, x+w//2,y+h//2) :
            #     print("Yes")
            #     claw=1
            # else :
            #     print("No")
            #     claw=0
            #sending coordinates to Arduino    
            #cv2.circle(image,(x+w//2,y+h//2),2,(0,255,0),2)
            string='X{0:d}Y{1:d}'.format((x+w//2),(y+h//2))
            print(string)
            ArduinoSerial.write(string.encode('utf-8')) 
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
      
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
      

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
     
          

    if cv2.waitKey(1) == ord("q"):
        break
      
# stop capturing
cv2.destroyAllWindows()
cap.release()