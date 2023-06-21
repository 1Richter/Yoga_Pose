# Yoga_Pose

## Directories

.CSV files in csv/test/ and csv/train/

trained with  [https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/download?datasetVersionNumber=1 ](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/download?datasetVersionNumber=1 )

## Files
csvwriter.py: input:images in data/TRAIN/ and data/TEST/ output:csv files in csv/train/ and csv/test/ with mediapipe
classify++.py: trains and tests the model with the csv files in csv/train/ and csv/test/ and saves the model in models/ with the name model.pt
test_mdl.py : Webcam to see poses live
