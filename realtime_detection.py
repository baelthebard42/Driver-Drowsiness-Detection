import cv2
import torch
import mediapipe as mp
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DrowsinessModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = './checkpoints/checkpoint_15.pth'
EYE_PROB_THRESHOLD = 0.35
YAWN_THREHOLD = 0.03
IN_CHANNELS = 3
OUT_CHANNELS = 4
FEATURES = [16, 32]
INPUT_SIZE = (145, 145)


def load_model():
    model = DrowsinessModel(in_channel=IN_CHANNELS, out_channel=OUT_CHANNELS, features=FEATURES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model_eyes = load_model()
model_face = load_model()


transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    
])


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

def preprocess(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0).to(DEVICE)

def predict(model, crop):
    tensor = preprocess(crop)
    with torch.no_grad():
        out = model(tensor)
        return torch.softmax(out, dim=1).cpu().numpy().squeeze()

def extract_eye_and_face(frame, landmarks, ih, iw):

    eye_indices = [33, 133, 159, 145]
    x_eye = [int(landmarks[i].x * iw) for i in eye_indices]
    y_eye = [int(landmarks[i].y * ih) for i in eye_indices]
    pad = 5
    eye_crop = frame[
        max(0, min(y_eye) - pad): min(ih, max(y_eye) + pad),
        max(0, min(x_eye) - pad): min(iw, max(x_eye) + pad)
    ]


    x_all = [int(lm.x * iw) for lm in landmarks]
    y_all = [int(lm.y * ih) for lm in landmarks]
    pad_face = 20
    face_crop = frame[
        max(0, min(y_all) - pad_face): min(ih, max(y_all) + pad_face),
        max(0, min(x_all) - pad_face): min(iw, max(x_all) + pad_face)
    ]

    return eye_crop, face_crop


while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "No Face Detected"
    color = (255, 255, 255)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        eye_crop, face_crop = extract_eye_and_face(frame, landmarks, ih, iw)

        if eye_crop.size != 0 and face_crop.size != 0:
            eye_probs = predict(model_eyes, eye_crop)
            face_probs = predict(model_face, face_crop)

            closed_prob = eye_probs[2]
            open_prob = eye_probs[3]
            yawn_prob = face_probs[0]
            no_yawn_prob = face_probs[1]

           
            if yawn_prob > YAWN_THREHOLD and closed_prob > EYE_PROB_THRESHOLD:
                status = "HIGH RISK! Driver likely going to fall asleep"
                color = (0, 0, 255)
            elif yawn_prob > YAWN_THREHOLD or closed_prob > EYE_PROB_THRESHOLD:
                status = "Warning: Drowsiness Detected"
                color = (0, 165, 255)
            else:
                status = "Fine"
                color = (0, 255, 0)

        
            cv2.putText(frame, f"Yawn: {yawn_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"No Yawn: {no_yawn_prob:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Closed: {closed_prob:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Open: {open_prob:.2f}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

  
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
