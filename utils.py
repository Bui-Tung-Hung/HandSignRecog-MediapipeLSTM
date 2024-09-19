import mediapipe as mp
def load_actions(filename='actions.txt'):
    import numpy as np
    with open(filename, 'r') as file:
        actions = [line.strip() for line in file.readlines()]
    return np.array(actions)


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
        