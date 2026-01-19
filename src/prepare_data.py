import os
import random
import shutil
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Raw_Drowsy =os.path.join(BaseDir, "data/raw/Drowsy/Drowsy")
Raw_non_Drowsy = os.path.join(BaseDir,"data/raw/Non_Drowsy/Non Drowsy")
proc_Drowsy = os.path.join(BaseDir,"data/processed/Drowsy")
proc_non_Drowsy=os.path.join(BaseDir,"data/processed/Non_Drowsy")
Sample_size = 2000
drowsy_files = [f for f in os.listdir(Raw_Drowsy) if f.lower().endswith(".png")]
nonDrowsyFiles= [ f for f in os.listdir(Raw_non_Drowsy) if f.lower().endswith(".png")]
print(len(drowsy_files))
print(len(nonDrowsyFiles))
sampledDrowsy = random.sample(drowsy_files, Sample_size)
sampledNonDrowsy = random.sample(nonDrowsyFiles, Sample_size)
for img in sampledDrowsy:
    src = os.path.join(Raw_Drowsy, img) #مسار الصورة الغير معالجة
    dst = os.path.join(proc_Drowsy,img) #مسار الصورة اللي بتكون فيها نكون مسار لكل صورة يعني
    shutil.copy(src, dst)
for img in sampledNonDrowsy:
    src = os.path.join(Raw_non_Drowsy, img)
    dst = os.path.join(proc_non_Drowsy, img)
    shutil.copy(src, dst)
print("Data preperation complete")
