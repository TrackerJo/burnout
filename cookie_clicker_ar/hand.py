from point import Point

class Hand:
    thumb = Point(0, 0)
    pointer = Point(0, 0)
    middle = Point(0, 0)
    min_dist = 10000
    cookies = 0
    
    def __init__(self, thumb, pointer, middle):
        self.thumb = thumb
        self.pointer = pointer
        self.middle = middle
        self.min_dist = 10000
        self.cookies = 0

