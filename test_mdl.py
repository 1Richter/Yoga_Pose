import cv2
import torch
from csvwriter import PoseDetector
from classify import Model

class ExercisePredictor:
    def __init__(self, model_path):
        self.detector = PoseDetector()
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def preprocess(self, img):
        position = self.detector.findPosition(img)
        flattened_position = []
        for id, x, y, z in position[0]:
            flattened_position.extend([x, y, z])
        tensor = torch.tensor(flattened_position, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, frame):
        tensor = self.preprocess(frame)
        prediction = self.model(tensor)
        pred_class = prediction.argmax(dim=1)
        return pred_class

if __name__ == "__main__":
    predictor = ExercisePredictor('model.pt')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        pred_class = predictor.predict(frame)
        print(f"Predicted exercise: {pred_class}")
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
