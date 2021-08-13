import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

def data_to_flatten(results):
  arr=np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
  print(len(arr))
  return arr

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image processing
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #dispalying the annotation in the image
    if results.multi_hand_landmarks:
      arr=data_to_flatten(results)
      print(arr)
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmark_style(),
            drawing_styles.get_default_hand_connection_style())


    # the exit function 
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
# print(results.multi_hand_landmarks)
print(len(results.multi_hand_landmarks[0].landmark))
cap.release()