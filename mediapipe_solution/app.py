import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def pre_process_landmark(landmark_list):

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        landmark_list[index][0] = np.abs(landmark_list[index][0] - base_x)
        landmark_list[index][1] = np.abs(landmark_list[index][1] - base_y)

    flattened = []
    for i in range(len(landmark_list)):
        flattened.append(landmark_list[i][0])
        flattened.append(landmark_list[i][1])


    normalized = []
    # Normalization
    for i in range(len(flattened)):
        normalized.append(flattened[i]/max(flattened))

    return normalized

def landmark_list(image, landmarks):
    height, width, _ = image.shape

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * width), width - 1)
        landmark_y = min(int(landmark.y * height), height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def prediction_to_str(pred):
    if pred+66 == ord(']'):
        return 'space'
    elif pred+65 == ord('['):
        return 'del'
    else:
        return chr(pred + 65)

def update_signed(pred, signed):
    if pred == 'del':
        signed = signed[:-1]
    elif pred == 'space':
        signed += ' '
    else:
        signed += pred
    
    return signed

def main():
    model_save_path = './Ricky/keypoint_classifier2.hdf5'
    model = tf.keras.models.load_model(model_save_path)
    signed = ''
    prediction = ''

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

    # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = landmark_list(image, hand_landmarks)
                    processed = pre_process_landmark(landmarks)
                    prediction = prediction_to_str(np.argmax(np.squeeze(model.predict(np.array([processed])))))
                    print(prediction)
                
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            cv2.putText(image, 'pred: '+prediction, (12,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)
            cv2.putText(image, signed, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(33) == ord(' '):
                signed = update_signed(prediction, signed)

    cap.release()

if __name__ == '__main__':
  main()
  