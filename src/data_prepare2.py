import os
import cv2
import numpy as np
from tqdm import tqdm
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_Drowsy = os.path.join(BaseDir, "data/processed/Drowsy")
proc_non_Drowsy = os.path.join(BaseDir, "data/processed/Non_Drowsy")
drowsy_images = [f for f in os.listdir(proc_Drowsy) if f.lower().endswith(".png")]
non_drowsy_images = [f for f in os.listdir(proc_non_Drowsy) if f.lower().endswith(".png")]
print("Drowsy images count: ", len(drowsy_images))
print("Non Drowsy images: ", len(non_drowsy_images))
X = []
Y = []
for img_name in tqdm(drowsy_images, desc="Processing Drowsy"):
    img_path = os.path.join(proc_Drowsy, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img / 225.0
    X.append(img)
    Y.append(1)
for img_name in tqdm(non_drowsy_images, desc="Processing non Drowsy"):
    img_path = os.path.join(proc_non_Drowsy, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    X.append(img)
    Y.append(0)
X = np.array(X)
Y = np.array(Y)
print("Final data set Shapes:")
print("X: ", X.shape)
print("Y: ", Y.shape)
np.save(os.path.join(BaseDir,"X.npy"),X)
np.save(os.path.join(BaseDir,"Y.npy"), Y)
print("Data saved successfully!")
