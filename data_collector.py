import cv2
import mediapipe as mp
import os
import numpy as np
mp_drawing =mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
drawing_styles=mp.solutions.drawing_styles

#function to convert the landmarks to single dimentional numpy array for lstm model tranning...
def data_to_flatten(results):
    arr=np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21*3)
    # print(len(arr))
    return arr
def dircreator():
    parrent_dir=os.getcwd()
    dir="data"
    seqnum=30
    actiondir=["rock","paper","scissors"]
    try:
        os.mkdir(dir)
    except OSError as error:
        #uncomment to debug
        # print(error)
        pass
    main_path=os.path.join(parrent_dir,dir)
    # print(main_path)
    for action in actiondir:
        try:
            namain=os.path.join(main_path,action)
            os.mkdir(namain)
        except OSError as error:
            pass
    for action in actiondir:
        namain=os.path.join(main_path,action)
        for i in range(seqnum):
            nsmain=os.path.join(namain,str(i))
            try:
                os.mkdir(nsmain)
            except OSError as error:
                pass
dircreator()

# variable declaration
parrent_dir=os.getcwd()
dir="data"
main_path=os.path.join(parrent_dir,dir)
seqnum=30
framenum=30
actiondir=["rock","paper","scissors"]

#data collection using opencv
cap=cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    for action in actiondir:
        for seq in range(seqnum):
            for frame in range(framenum):
                # reading the image from camera
                success,image=cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                #rotating the image aroung y-axis and changing the color channel to rgb
                image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
                image.flags.writeable=False

                results=hands.process(image)
                image.flags.writeable=True
                image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmark_style(),
                            drawing_styles.get_default_hand_connection_style())
                else:
                    cv2.putText(image, 'Hand is not there in a frame', (120,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA) 
                if frame == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, seq), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, seq), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                # Break gracefully
                data=data_to_flatten(results)
                npy_path=os.path.join(main_path,action,str(seq),str(frame))
                np.save(npy_path,data)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()

        