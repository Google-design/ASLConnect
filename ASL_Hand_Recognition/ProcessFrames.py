import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import base64
import io

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

# Load the model
model = ASLCNN().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(torch.load('asl_cnn_model_mnist_ABD.pth', map_location=torch.device('cpu')))
model.eval()

# MediaPipe hands setup
mp_hands = mp.solutions.hands.Hands(max_num_hands=1)

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode_image(image_data):
    """Decode base64 image to numpy array."""
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def preprocess_hand_image(hand_image):
    """ Convert image to grayscale and apply transformations """
    hand_image = transform(hand_image).unsqueeze(0)  # Add batch dimension
    return hand_image

def extract_hand_image(frame, hand_landmarks):
    """ Extract hand region from the frame """
    h, w, _ = frame.shape
    x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
    x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
    y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
    y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)
    return frame[y_min:y_max, x_min:x_max]

def process_frame(image_data, timestamp):
    """ Process the image to detect hands and classify the hand pose """
    image = decode_image(image_data)
    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_img = extract_hand_image(image, hand_landmarks)
            if hand_img.size > 0:  # Ensure hand_img is not empty
                hand_img_tensor = preprocess_hand_image(hand_img)
                with torch.no_grad():
                    prediction = model(hand_img_tensor)
                    predicted_class = torch.argmax(prediction, dim=1).item()
                    return f"ASL: {chr(predicted_class + 65)}"  # Convert class number to letter

    return "No hand detected"
