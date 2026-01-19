import os
import cv2
import matplotlib.pyplot as plt
import random 
import numpy as np
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_Drowsy = os.path.join(BaseDir, "data/processed/Drowsy")
proc_non_Drowsy = os.path.join(BaseDir, "data/processed/Non_Drowsy")
drowsyImages = [f for f in os.listdir(proc_Drowsy) if f.lower().endswith(".png")]
nonDrowsyImages = [f for f in os.listdir(proc_non_Drowsy) if f.lower().endswith(".png")]
sampleDrowsy = random.choice(drowsyImages)
sampleNonDrowsy = random.choice(nonDrowsyImages)
drowsyImg = cv2.imread(os.path.join(proc_Drowsy, sampleDrowsy))
nonDrowsyImg = cv2.imread(os.path.join(proc_non_Drowsy, sampleNonDrowsy))
drowsyImg = cv2.cvtColor(drowsyImg, cv2.COLOR_BGR2RGB)
nonDrowsyImg = cv2.cvtColor(nonDrowsyImg, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(drowsyImg)
plt.title("Drowsy")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(nonDrowsyImg)
plt.title("Non Drowsy")
plt.axis("off")

plt.show()

drowsyArray= drowsyImg / 255.0
nonDrowsyArray = nonDrowsyImg / 255.0

print("Drowsy images shape: " ,drowsyArray.shape)
print("Non Drowsy images shape: " ,nonDrowsyArray.shape)