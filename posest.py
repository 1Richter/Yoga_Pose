import cv2
import mediapipe as mp
import time
import numpy as np
# from schoettls import MyDataset
# from schoettls import demo2 as demo_print
from torch.utils.data import Dataset, DataLoader
from os import path, listdir
import matplotlib.pyplot as plt

root_dir = path.dirname(path.abspath(__file__))

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)  # Convert to degrees within 360 degrees
    if angle > 180.0:
        angle = 360 - angle

    return angle


class PoseDetector:

    def __init__(self, mode=False, upBody=1, smooth=False, detectionCon=True, trackCon=0.6, f=0.75):
        self.results = None
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.f = f

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon, self.f)
        self.counter = 0
        self.stage = None

    def findPose(self, img, draw=True, counter=0):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        try:
            landmarks = self.results.pose_landmarks.landmark
            shoulder = [landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            # visualize
            cv2.putText(img, str(int(angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Counter
            if angle > 130:
                self.stage = "down"
                print(self.counter, self.stage)
            elif int(angle) < 50 and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                print(self.counter, self.stage)

        except:
            pass
        #render counter
        cv2.rectangle(img, (350, 0), (600, 60), (245, 117, 16), -1)
        cv2.putText(img, 'Counter', (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(self.counter), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    
    def get_ankle_positions(self, img):
        lmList = self.findPosition(img, draw=False)
        if len(lmList) > 0:
            left_ankle = lmList[self.mpPose.PoseLandmark.LEFT_ANKLE.value][:2]
            right_ankle = lmList[self.mpPose.PoseLandmark.RIGHT_ANKLE.value][:2]
            return left_ankle, right_ankle
        return None


def write_csv(position, print_lm_name=True):
    import csv
    file = root_dir + "/positions.csv"
    with open(file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        flattened_position = []
        for id, x, y in position[0]:
            if print_lm_name:
                landmark_name = mp.solutions.pose.PoseLandmark(id).name
            else:
                landmark_name = id
            flattened_position.extend([x, y])
        csvwriter.writerow(flattened_position)
        csvfile.flush() # flush the buffer to file

def show_only_lm(img, position):
    # Create a blank image with the same dimensions as the original image
    h, w, c = img.shape
    blank_img = np.zeros((h, w, c), dtype=np.uint8)* 255

    # Draw the landmark points on the blank image
    if position:
        for _, x, y in position:
            cv2.circle(blank_img, (x, y), 5, (255, 0, 0), cv2.FILLED)
    # Draw the connections on the blank image
        # if detector.results.pose_landmarks:
        #     detector.mpDraw.draw_landmarks(blank_img, detector.results.pose_landmarks, detector.mpPose.POSE_CONNECTIONS, None, None, (0, 0, 255))


    cv2.imshow("Image Tracker", blank_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main(image_path = None):
    if image_path:
        trackCon_values = np.linspace(0.01, .99, 5)  # Adjust as needed
        detectCon_values = np.linspace(.05, .96, 4)  # Adjust as needed
       

        img = cv2.imread(image_path)
        if img is None:
            print("Image is None")
            return

        fig, axs = plt.subplots(len(detectCon_values), len(trackCon_values), figsize=(20, 20))
        fig.suptitle('Pose Detection with Varying trackCon and detectCon Values')

        for i, detectCon in enumerate(detectCon_values):
            for j, trackCon in enumerate(trackCon_values):
                detector = PoseDetector(0, 1, False, True, trackCon, f=detectCon)
                img_pose = detector.findPose(img.copy())  # Copy the image to avoid modifying the original
                axs[i, j].imshow(cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB))  # Convert color space for matplotlib
                axs[i, j].axis('off')  # Hide axes for clarity
                axs[i, j].set_title(f'trackCon={trackCon:.2f}, detectCon={detectCon:.2f}')

        plt.tight_layout()
        plt.show()
    # if image_path:
    #     detector = PoseDetector()
    #     positions = []

    #     img = cv2.imread(image_path)
    #     if img is None:
    #         print("Image is None")
    #         return
    #     img = detector.findPose(img)
    #     position = detector.findPosition(img, draw=False)
        

    #     if position:
    #         positions.append(position)
    #         #show_only_lm(img, position)
    #         write_csv(positions, print_lm_name=True)


    #     cv2.imshow("Image Tracker", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     cap = cv2.VideoCapture(0)
    #     pTime = 0
    #     detector = PoseDetector()
    #     positions = []

    #     while cap.isOpened():
    #         success, img = cap.read()
    #         img = detector.findPose(img)
    #         # lmList = detector.findPosition(img)
    #         # if len(lmList) != 0:
    #         #     print(lmList[14])
    #         position = detector.findPosition(img, draw=False)
    #         if position:
    #             positions.extend(position)

    #         cTime = time.time()
    #         fps = 1 / (cTime - pTime)
    #         pTime = cTime
    #         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    #         cv2.imshow("Webcam Tracker", img)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
            
    #     write_csv(positions)



if __name__ == "__main__":
    # now only reference file path and not the file image itself
    # iterate over every file in the directory /data/TRAIN/downdog
    i = 1
    for file in listdir(root_dir + "/data/TRAIN/downdog"):
        if file.endswith(".jpg"):
            if i % 4 == 0:
                data = root_dir + "/data/TRAIN/downdog/" + file
                main(data)
            i += 1

""" extract data from csv
import pandas as pd
import torch

def load_ankle_positions(file_name):
    df = pd.read_csv(file_name)
    ankle_positions = torch.tensor(df.values, dtype=torch.float32)
    return ankle_positions

ankle_positions = load_ankle_positions('ankle_positions.csv')
print(ankle_positions)
"""
