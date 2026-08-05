import math

class Helpers:
    def dist_between_points(p1, p2):
        return math.sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2))