import cv2
import torch
from csvwriter import PoseDetector
from classifypp import Model, DataHandler
import mediapipe as mp

class ExercisePredictor:
    def __init__(self, model_path):
        self.detector = PoseDetector()
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.classes = DataHandler(None, None).class_to_label
        self.results = None
        self.frames = []
        self.pred_class = "None"


    def predict(self, frame):
        landmarks = self.detector.findPosition(frame)
        # print("predict", end = " ")
        if landmarks:
            # Flatten landmarks into a single list
            flattened_landmarks = [coord for landmark in landmarks for coord in landmark]
            flattened_landmarks = torch.tensor(flattened_landmarks)

            self.frames.append(flattened_landmarks)

        if len(self.frames) == (batch_size := 32): 
            batch = torch.stack(self.frames)
            output = self.model(batch)
            from collections import Counter
            most_common = Counter(output.argmax(dim=1).tolist()).most_common(1)[0][0]
            self.pred_class = [k for k,v in self.classes.items() if v == most_common][0]
            self.frames = []
            print(self.pred_class)
            return self.pred_class
        else:
            # wait for the next frame
            return self.pred_class

        

    def print_lm_on_screen(self, img, text = None):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.detector.pose.process(imgRGB)
        if self.results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            cv2.putText(img, str(text), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img


if __name__ == "__main__":
    predictor = ExercisePredictor('model.pt')
    cap = cv2.VideoCapture(0)
    pTime = 0
    positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        pred_class = predictor.predict(frame)
        frame = predictor.print_lm_on_screen(frame, text=pred_class)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
