import  cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from collections import  deque
import time
import os
import pyautogui

import threading
labels = ['SwipeR','SwipeL','none','clap']
seqL = 150
numseq = 150
def landmarkgetter(sequence):
        data = np.zeros(126, dtype=np.float32)
        if sequence.hand_landmarks:
            for i , hand in enumerate(sequence.hand_landmarks[:2]):
                ofst = i * 63
                for j, lm in enumerate(hand):
                    data[ofst+j * 3+0]=lm.x
                    data[ofst+j * 3+1]=lm.y
                    data[ofst+j * 3+2]=lm.z
        return data
class lstm(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        return self.fc2(x)
def realtimegestureRec():
    MODEL_DIR = 'models'
    modelp = os.path.join(MODEL_DIR, "gesture_lstm.pt")
    labelsp = os.path.join(MODEL_DIR, "labels.npy")
    sql = 15
    holdtime = 0.2
    waittine = 2
    threshs = {'clap': 0.5, 'SwipeL': 0.5, 'SwipeR': 0.5}
    
    labellist = list(np.load(labelsp, allow_pickle=True))
    nclass = len(labellist)
    device = torch.device("cpu")
    model = lstm(input_size=126, num_classes=nclass).to(device)
    model.load_state_dict(torch.load(modelp, map_location=device))
    model.eval()
    base = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    seq = deque(maxlen=sql)
    predlabel = ''
    predconf = 0.0
    predtime = 0
    last_triggered = {
    'clap': 0.0,
    'SwipeL': 0.0,
    'SwipeR': 0.0
    }
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        now = time.time()
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        data = landmarkgetter(result)
        seq.append(data)
        if len(seq) == sql:
            with torch.no_grad():
                input_data = torch.from_numpy(np.array(seq)).float().unsqueeze(0).to(device)
                out = model(input_data)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                label = labellist[pred]
            if label in threshs and conf >= threshs[label]:
                if now - last_triggered[label] >= waittine:
                    last_triggered[label] = now
                    predlabel = label
                    predconf = conf
                    predtime = now
                    seq.clear()
                    if label == 'clap':
                        threading.Thread(
    target=lambda: os.system('ffplay -nodisp -autoexit clapping.mp3'),
    daemon=True
).start()
                    elif label == 'SwipeL':
                        pyautogui.press('left')
                        
                    elif label == 'SwipeR':
                        pyautogui.press('right')
                        
        if predlabel != '':
            cv2.putText(frame, f'{predlabel} {predconf*100:.1f}%', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            if time.time() - predtime > holdtime:
                predlabel = ''
        for i, (g, last) in enumerate(last_triggered.items()):
            remaining = max(0.0, waittine - (now - last))
            status = f"{g}: ready" if remaining == 0 else f"{g}: {remaining:.0f}s"
            cv2.putText(frame, status, (10, frame.shape[0] - 20 - i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
realtimegestureRec()