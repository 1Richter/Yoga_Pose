import cv2
import torch
from csvwriter import PoseDetector
from classifypp import Model

class ExercisePredictor:
    def __init__(self, model_path):
        self.detector = PoseDetector()
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.results = None

    def preprocess(self, img):
        position = self.detector.findPosition(img)
        flattened_position = []
        for x, y, z in position[0]:
            flattened_position.extend([x, y, z])
        tensor = torch.tensor(flattened_position, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, frame):
        tensor = self.preprocess(frame)
        prediction = self.model(tensor)
        pred_class = prediction.argmax(dim=1)
        return pred_class

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
        pred_class = predictor.predict(frame)
        frame = predictor.print_lm_on_screen(frame, text=pred_class)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
