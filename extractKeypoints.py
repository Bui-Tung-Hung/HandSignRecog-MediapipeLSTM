import cv2
import numpy as np
import os
import mediapipe as mp
from utils import *

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results




def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=0, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,255,120), thickness=1, circle_radius=1))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,255,120), thickness=1, circle_radius=1))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,255,120), thickness=1, circle_radius=1))
        





def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])




# Path for exported data
DATA_PATH = os.path.join('MP_Data')  # Folder for .npy files
VIDEO_PATH = os.path.join('MP_Videos')  # Folder for videos


# Create directories if not exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(VIDEO_PATH, exist_ok=True)



# Actions that we try to detect ## Chọn hành động để thống nhất
actions = load_actions('actions.txt')


## Thay đổi parameters để thêm nhiều data, tăng độ chính xác
# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30



#Make dirs
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
        os.makedirs(os.path.join(VIDEO_PATH, action, str(sequence)), exist_ok=True)






cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object (MJPG format)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            # Define the output video path and create VideoWriter
            video_path = os.path.join(VIDEO_PATH, action, str(sequence), f"{action}_{sequence}.avi")
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))  # Assuming 640x480 resolution

            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Write the frame to the video
                out.write(image)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            # Release the video writer after finishing the sequence
            out.release()

    cap.release()
    cv2.destroyAllWindows()
