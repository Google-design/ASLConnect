import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import csv

# Define the CNN model with batch normalization
class ASLCNN(nn.Module):
    def __init__(self):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 26)  # 26 output classes (A-Z)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d(2, 2)(x)
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

class VideoCamera:
    def __init__(self):
        # Initialize the camera and model
        self.video = cv2.VideoCapture(0)
        self.model = ASLCNN().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(torch.load('asl_cnn_model_mnist_ABD.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Extract hand region
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size != 0:
                    hand_img = self.preprocess_hand_image(hand_img)
                    prediction = self.model(hand_img)
                    predicted_class = torch.argmax(prediction, dim=1).item()
                    cv2.putText(frame, f'ASL: {chr(predicted_class + 65)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def preprocess_hand_image(self, hand_image):
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
        hand_image = np.array(hand_image, dtype=np.uint8)
        hand_image = self.transform(hand_image).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return hand_image
    
if __name__ == "__main__":
    cam = VideoCamera()
    while True:
        frame = cam.get_frame()
        if frame is None:
            break
        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("ASL Recognition", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.__del__()
    cv2.destroyAllWindows()
