from point import Point
from helpers import Helpers
class Hand:
    thumb = Point(0, 0)
    pointer = Point(0, 0)
    middle = Point(0, 0)
    min_dist = 10000
    team = 0
    onClick = any
    is_clicked = False
    
    def __init__(self, thumb, pointer, middle, onClick):
        self.thumb = thumb
        self.pointer = pointer
        self.middle = middle
        self.min_dist = 10000
        self.onClick = onClick

    def process(self):
        dist = Helpers.dist_between_points(self.pointer, self.thumb)
        self.min_dist = min(self.min_dist, dist)
        if dist >= 30 and self.min_dist <= 25:
            self.min_dist = dist
            self.is_clicked = False
            self.onClick(self)
        if dist <= 25:
            self.is_clicked = True


