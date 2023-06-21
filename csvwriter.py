import cv2
import csv
from  mediapipe import solutions
from os import path, listdir, sep
import mediapipe as mp
root_dir = path.dirname(path.abspath(__file__))

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=True, trackCon=0.6, f = 0.75):
        self.results = None
        self.pose = solutions.pose.Pose(mode, upBody, smooth, detectionCon, trackCon, f)

    def findPosition(self, img, absolute_coord= False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                if absolute_coord:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                else:
                    cx, cy, cz = lm.x, lm.y, lm.z
                lmList.append([cx, cy, cz])
        return lmList

def write_csv(position, file_name= "t", print_lm_name=True):
    file = path.join(root_dir, "csv", file_name + ".csv")
    with open(file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        flattened_position = []
        for x, y, z in position[0]:
            # if print_lm_name:
            #     landmark_name = solutions.pose.PoseLandmark(id).name
            # else:
            #     landmark_name = id
            flattened_position.extend([x, y, z])
            



        csvwriter.writerow(flattened_position)
        csvfile.flush()  # flush the buffer to file


    

def main(image_path = None):
    if image_path:
        detector = PoseDetector()
        img = cv2.imread(image_path)
        if img is None:
            print("Image is None")
            return
        position = detector.findPosition(img)
        if position:
            directories = path.dirname(image_path).split(sep)[-2:]
            file= directories[1] + "_" + path.basename(image_path)
            file_name = path.join(root_dir, "csv", directories[0], file)
            write_csv([position], file_name, print_lm_name=False)

if __name__ == "__main__":
    train_dir = path.join(root_dir, "data\\TEST")
    for directory in listdir(train_dir):
        print(directory)
        for file in listdir(path.join(train_dir, directory)):
            if file.endswith(".jpg"):
                data=path.join(train_dir, directory, file)
                main(data)
