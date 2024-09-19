import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from mediapipe import solutions as mp_solutions
from utils import * 


# Đường dẫn tới thư mục chứa dữ liệu
DATA_PATH = os.path.join('MP_Data')

# Các hành động sẽ được nhận diện (actions)
actions = load_actions("actions.txt")

# Tải mô hình đã huấn luyện
model = load_model('action_recognition_model.h5')

# Khởi tạo Mediapipe để nhận diện keypoints
mp_holistic = mp_solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp_solutions.drawing_utils

# Hàm xử lý video với Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Hàm vẽ các keypoints
draw_styled_landmarks(image, results)

# Hàm trích xuất keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Khởi tạo các biến để lưu chuỗi hành động
sequence = []
sentence = []
predictions = []
threshold = 0.8

# Sử dụng webcam để nhận diện hành động
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Phát hiện keypoints bằng Mediapipe
    image, results = mediapipe_detection(frame, mp_holistic)

    # Vẽ keypoints trên video
    draw_styled_landmarks(image, results)

    # Trích xuất keypoints và lưu vào sequence
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Chỉ giữ lại 30 frame gần nhất

    # Dự đoán hành động nếu đủ 30 frame
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))

        # Nếu độ tin cậy của dự đoán lớn hơn ngưỡng, thêm hành động vào câu
        if np.max(res) > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        # Giới hạn độ dài của câu
        if len(sentence) > 5:
            sentence = sentence[-5:]

    # Hiển thị kết quả dự đoán trên màn hình
    cv2.putText(image, ' '.join(sentence), (3, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Hiển thị video
    cv2.imshow('OpenCV Feed', image)

    # Thoát chương trình khi nhấn phím 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
