import os
drowsy_path = "data/raw/Drowsy/Drowsy"
non_drowsy_path = "data/raw/Non_Drowsy/Non Drowsy"
drowsy_images = os.listdir(drowsy_path)
non_drowsy_images = os.listdir(non_drowsy_path)
print("Drowsy Images: ", len(drowsy_images))
print("Non Drowsy Images: ", len(non_drowsy_images))