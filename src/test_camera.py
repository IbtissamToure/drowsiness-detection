import cv2
import numpy as np
import tensorflow as tf
from collections import deque

frame_count = 0
face_cascade= cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml" )
model = tf.keras.models.load_model("drowsiness_model.h5")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pred_buffer = deque(maxlen=15)
drowsy_counter = 0
drowsy_threshold = 0.25
drowsy_frames_needed = 15
labels = ["Alert" , "Drowsy"]
last_pred = "Alert"

faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count+=1
    if frame_count % 5 == 0:
        small = cv2.resize(frame, None, fx = 0.5, fy= 0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces = [(x*2, y*2, w*2, h*2) for (x,y,w,h) in faces]
    

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        pred_buffer.clear()
        drowsy_counter = 0
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]

    if frame_count % 10 == 0:
       img = cv2.resize(face,(224,224))
       img = img/255.0
       img = np.expand_dims(img, axis=0)

       pred = model.predict(img)[0][0]
       pred_buffer.append(pred)
       avg_pred = sum(pred_buffer) / len(pred_buffer)

       if avg_pred > drowsy_threshold :
           drowsy_counter = min(drowsy_counter +1,  drowsy_frames_needed) 
       else:  
           drowsy_counter = max(drowsy_counter -1 , 0)
           
       
       index = 1 if drowsy_counter >= drowsy_frames_needed else 0
       last_pred = labels [index]

    
    cv2.putText(frame, last_pred, (20,40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,255,0), 2
    )
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
