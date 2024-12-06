import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageFont, ImageDraw

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class Draw():
    """ Helper class for drawing utilities """
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.font = ImageFont.truetype('arial.ttf', self.height//24, encoding="unic")

    def bbox(self):
        """ Draw bbox """
        pass

    def skeleton(self, image, pose_results):
        """ Draw skeleton with pose landmarks """
        mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return image

    def draw_line(self, image, coord1, coord2):
        """ Draw a line in image """
        cv2.line(image, coord1, coord2, thickness=4, color=(255, 255, 255))
        return 
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("human-pose-estimation-opencv/dance.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    draw_utils = Draw(width,height)

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame = draw_utils.skeleton(frame, results)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

