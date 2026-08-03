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

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SHOW_LANDMARKS = False

THUMB = 4
POINTER_FINGER = 8
MIDDLE_FINGER = 12

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

def dist_between_points(p1, p2):
    return math.sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2))

def draw_hand_landmarks(img, hand_landmarks, width, height, hand_idx):
    """Draw a dot for every landmark, and a bigger highlighted dot on the index fingertip."""
    global hands
    while len(hands) <= hand_idx:
        hands.append(Hand(Point(-1, -1), Point(-1, -1),Point(-1, -1)))
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


def draw_cookies(img):
    global hands
    for hand in hands:
        cv2.putText(img, f'Cookies: {int(hand.cookies)}', (hand.middle.x, hand.middle.y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

rectangle = Rectangle(Point(120, 250), 150, 150, "cookie.png")
hands = []


def main():
    global hands
    download_model_if_needed()
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_timestamp_ms = 0
    in_cookie = False

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
        clicked = False
        for hand in hands:
            dist = dist_between_points(hand.pointer, hand.thumb)
            in_cookie = rectangle.inRectangle(hand.pointer)
            if in_cookie:
                hand.min_dist = min(hand.min_dist, dist)
                if dist >= 30 and hand.min_dist <= 25:
                    hand.cookies += 1
                    hand.min_dist = dist
                    clicked = clicked or False
                if dist <= 25:
                    clicked = True
        rectangle.draw(cv2, frame, clicked)
        draw_cookies(frame)
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()