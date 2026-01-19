# Drowsiness Detection Project

A computer vision project to detect driver drowsiness using deep learning. The goal is to experiment with image-based detection while understanding model limitations.


## Project Overview

This project aims to detect drowsiness from video frames. The main objectives are:

- **Data preprocessing:** Prepare datasets for training.
- **Model training:** Train a deep learning model using MobileNetV2 features.
- **Evaluation:** Measure accuracy and visualize results.
- **Camera testing:** Real-time testing using a webcam.

> Note: Model accuracy may vary depending on data quality and environment.


## Project Structure
cv-project/
├─ src/
│ ├─ init.py
│ ├─ data_check.py
│ ├─ data_prepare2.py
│ ├─ prepare_data.py
│ ├─ test_camera.py
│ ├─ train.py
│ └─ visualize_data.py
├─ notebooks/
│ └─ exploration.ipynb
├─ tests/
│ └─ test_model.py
├─ utils/
│ └─ helpers.py
├─ requirements.txt
├─ .gitignore
└─ README.md

> Large datasets (`data/`) and trained model weights (`*.h5`) are **excluded** from the repository due to size and privacy reasons.

## Setup Instructions

1. Clone the repository:

git clone https://github.com/username/repo-name.git
cd repo-name

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run any script from src/ or notebooks/ to start exploring the project.

## Notes:

Images and model weights are not included due to size constraints.

The project is designed for educational purposes and experimentation.

Accuracy may vary depending on environment and dataset quality.

## Author: 
Ibtissam Toure