import random

import cv2
import mediapipe as mp
import time
import urllib.request
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from rectangle import Rectangle
from point import Point
from hand import Hand
from helpers import Helpers

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SHOW_LANDMARKS = False

THUMB = 4
POINTER_FINGER = 8
MIDDLE_FINGER = 12

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def download_model_if_needed():
    """Download the hand landmark model once, if it's not already saved locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def create_detector():
    """Set up and return the MediaPipe HandLandmarker."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=6,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)

def draw_hand_landmarks(img, hand_landmarks, width, height, hand_idx):
    """Draw a dot for every landmark, and a bigger highlighted dot on the index fingertip."""
    global hands
    while len(hands) <= hand_idx:
        hands.append(Hand(Point(-1, -1), Point(-1, -1),Point(-1, -1), onClick))
    for id, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * width), int(lm.y * height)

        if id == POINTER_FINGER:
            if SHOW_LANDMARKS: 
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            hands[hand_idx].pointer = Point(cx, cy, lm.z)
        elif id == THUMB:  
            if SHOW_LANDMARKS:
                cv2.circle(img, (cx, cy), 3, (255, 255, 0), cv2.FILLED)
            hands[hand_idx].thumb = Point(cx, cy, lm.z)
        elif id == MIDDLE_FINGER:
            hands[hand_idx].middle = Point(cx, cy, lm.z)
        else:
            if SHOW_LANDMARKS:
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)


def draw_points(img):
    global hands
    global left_points
    global right_points
    for hand in hands:
        cv2.putText(img, f'Points: {left_points if hand.team == 0 else int(right_points)}', (hand.middle.x, hand.middle.y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 255) if hand.team == 1 else (0, 255, 0), 2)
hands = []
cookie_objs = []
frame_timestamp_ms = 0
left_points = 0
right_points = 0

def draw_objects(frame):
    global cookie_objs
    for obj in cookie_objs:
        obj.draw(cv2, frame)

def move_objects():
    global cookie_objs
    global left_points
    global right_points
    cookies_rem = []
    for obj in cookie_objs:

        obj.center.y += obj.speed if not obj.flip else -obj.speed 
        if obj.center.y + (obj.height / 2) > FRAME_HEIGHT:
            cookies_rem.append(obj)
            if obj.center.x > FRAME_WIDTH / 2:
                min_x = 75
                max_x =  FRAME_WIDTH / 2 + 75 / 2 
                x = random.uniform(min_x, max_x)
                cookie_objs.append(Rectangle(Point(x, 75), 75, 75, "cookie.png", 75))
                for hand in hands:
                    if hand.team == 0:
                        left_points += 1
            else:
                min_x = FRAME_WIDTH / 2 + 75 / 2 
                max_x = FRAME_WIDTH - 75 / 2
                x = random.uniform(min_x, max_x)
                cookie_objs.append(Rectangle(Point(x, 75), 75, 75, "cookie.png", 75))
                for hand in hands:
                    if hand.team == 1:
                        right_points += 1
        if obj.center.y - (obj.height / 2) <= 0:
            obj.flip = False
            if obj.center.x > FRAME_WIDTH / 2:
                min_x = 75
                max_x =  FRAME_WIDTH / 2 + 75 / 2 
                x = random.uniform(min_x, max_x)
                obj.center.x = x
            else:
                min_x = FRAME_WIDTH / 2 + 75 / 2 
                max_x = FRAME_WIDTH - 75 / 2
                x = random.uniform(min_x, max_x)
                obj.center.x = x
            obj.center.y = 75
            obj.speed += 2
    for obj in cookies_rem:
        cookie_objs.remove(obj)

def onClick(hand):
    global cookie_objs

    cookies_rem = []
    if hand.team == 0:
        for obj in cookie_objs:
            if obj.inRectangle(hand.pointer):
                obj.flip = True
                break
        for obj in cookies_rem:
                cookie_objs.remove(obj)
    else:
        for obj in cookie_objs:
            if obj.inRectangle(hand.pointer):
                obj.flip = True
                break
        for obj in cookies_rem:
                cookie_objs.remove(obj)
    
    

def draw_screen(frame):
    draw_objects(frame)
    draw_points(frame)
    cv2.imshow("Image", frame)

def init():
    min_x = FRAME_WIDTH / 2 + 75 / 2 
    max_x = FRAME_WIDTH - 75 / 2
    x = random.uniform(min_x, max_x)
    cookie_objs.append(Rectangle(Point(x, 75), 75, 75, "cookie.png", 75))

def main():
    global hands
    global frame_timestamp_ms
    download_model_if_needed()
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    
    init()
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        frame_timestamp_ms += 1
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        height, width, _ = frame.shape
        if result.hand_landmarks:
            hands = hands[:len(result.hand_landmarks)]
            for hand_idx,hand_landmarks in enumerate(result.hand_landmarks):
                draw_hand_landmarks(frame, hand_landmarks, width, height, hand_idx)
        for hand in hands:
            hand.process()
            if hand.pointer.x < FRAME_WIDTH/2:
                hand.team = 0
            else:
                hand.team = 1
        move_objects()
        
        draw_screen(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()