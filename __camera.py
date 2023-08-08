import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import numpy as np
import time
import dlib
import cv2
import os
import math
import face_recognition
import pickle
from collections import deque
import pandas as pd
from datetime import datetime
import mediapipe as mp
import requests
import psutil

class VideoCamera(object):
    def __init__(self):
   
        url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"
        self.video = WebcamVideoStream(src=0).start()
        print("Thread Started")
        threads_count = psutil.cpu_count() / psutil.cpu_count(logical=False)
        print("No:"+str(threads_count))
     
        self.known_face_encodings=[]
        self.known_face_names=[]
        if not os.path.exists("../face_rec/encodings.pkl"):
            my_list = os.listdir('../face_rec/Training_images')
            for i in range(len(my_list)):
                if(my_list[i]!=".ipynb_checkpoints"):
                    image=face_recognition.load_image_file("../face_rec/Training_images/"+my_list[i])
                    print(my_list[i])
                    face_encoding = face_recognition.face_encodings(image,num_jitters=100)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(my_list[i])

            with open('../face_rec/encodings.pkl','wb') as f:
                pickle.dump([self.known_face_encodings,self.known_face_names], f)
        else:
            with open('../face_rec/encodings.pkl', 'rb') as f:
                self.known_face_encodings ,self.known_face_names = pickle.load(f)
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(model_selection=1)
        
        
        self.distances = []

        self.tTime = 0.0
        self.pTime = 0
        self.timer = 0.0
        self.isRequest = False
        self.users={}
        
    def __del__(self):
        #self.video.release()
        self.video.stopped=True

    def picture_from_frame(self,frame,name = "unknown", confidence=0.0):
        this_time = datetime.now().isoformat(timespec='minutes')
        known_dir="../face_rec/captured_known_images/"+name
        # cap = gen_capture(url=0)
        if not (os.path.isdir(known_dir)):
            mode = 0o777
            os.makedirs(known_dir,mode)
        
        file_path = known_dir+'/'+this_time+'_'+str(confidence)+'.jpg'
        # print()
        cv2.imwrite(file_path,frame)
        return file_path

    def get_frame(self):
        rgb_frame = self.video.frame
        if rgb_frame is not None:
            rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.4, fy=0.3333)

        results = self.faceDetection.process(rgb_frame)
        face_locations = []
    
        if results.detections:            
            self.timer = time.time()
            if self.tTime == 0.0:
                self.tTime = time.time()
            for id,detection in enumerate(results.detections):
                bBoxC=detection.location_data.relative_bounding_box
                ih,iw,ic=rgb_frame.shape
                bBox = int(bBoxC.xmin*iw),int(bBoxC.ymin*ih),int(bBoxC.width*iw),int(bBoxC.height*ih)
                left,top,right,bottom = bBox[1],bBox[0]+bBox[2],bBox[1]+bBox[3],bBox[0]
                tup=(left,top,right,bottom)
                face_locations.append(tup)

        cTime = time.time()
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime
        
        cv2.putText(rgb_frame, "FPS: {:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        faces = []
        count = 0
        dTime = time.time()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # count+=1         
            print(face_locations)
            
            #single face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            
            confidence=min(face_distances)
            
            if min(face_distances) < 0.50: 
        
                matchIndex = np.argmin(face_distances)
                if matches[matchIndex]:
                    
                    name = self.known_face_names[matchIndex].upper()
                    if name not in self.users:
                        self.users[name]=1
                    else:
                        self.users[name]=self.users[name]+1
                    

                    #print(self.users)
                    print(confidence)
                        
                print(self.users)
                print(name)
                
      
                just_name = name.split('-')[0]
                cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0,255,0), 1)
                
                cv2.putText(rgb_frame, just_name, (left , bottom + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (0, 255, 0), 1)
        
        ret, jpeg = cv2.imencode('.jpg', rgb_frame)
        return jpeg.tobytes()
