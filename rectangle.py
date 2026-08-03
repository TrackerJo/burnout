from point import Point
import numpy as np


class Rectangle:
    center = None
    width = 0
    height = 0
    image = ""

    def __init__(self, center, width, height, image=""):
        self.center = center
        self.width = width
        self.height = height
        self.image = image

    def topLeft(self):
        return Point(self.center.x - self.width / 2, self.center.y - self.height / 2)
    
    def topRight(self):
        return Point(self.center.x + self.width / 2, self.center.y - self.height / 2)

    def bottomRight(self):
        return Point(self.center.x + self.width / 2, self.center.y + self.height / 2)
    

    def bottomLeft(self):
        return Point(self.center.x - self.width / 2, self.center.y + self.height / 2)

    def inRectangle(self, point):
         if point.x == -1 and point.y == -1: 
             return False
         return self.topLeft().x <= point.x and self.topLeft().y <= point.y and self.bottomRight().x >= point.x and self.bottomRight().y >= point.y

    def draw(self, cv2, frame, filled):
        top_left = self.topLeft()
        bottom_right = self.bottomRight()
        
        # cv2.rectangle(
        #     frame,
        #     (int(top_left.x), int(top_left.y)),
        #     (int(bottom_right.x), int(bottom_right.y)),
        #     (0, 255, 0),
        #     -1 if filled else 2 
        # )

        if self.image != "":
            logo = cv2.imread(self.image)
            size = 140 if filled else 150
            logo = cv2.resize(logo, (size, size))
            img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            success, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            x1 = int(self.center.x - size / 2)
            y1 = int(self.center.y - size / 2)
            roi = frame[y1:y1 + size, x1:x1 + size]
        
            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo