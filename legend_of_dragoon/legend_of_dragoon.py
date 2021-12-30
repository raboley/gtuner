import os
import cv2

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.gcvdata = bytearray([0x00, 0x00, 0x00, 0x00, 0x00])
        self.overworld_template = cv2.imread("Images/overworld_height_thing.jpg")
        self.item_full_template = cv2.imread("Images/too_many_items_replace.jpg")
        self.item_menu_template = cv2.imread("Images/items_menu.jpg")
        self.healing_rain = cv2.imread("Images/healing_rain.jpg")
        self.sort_menu = cv2.imread("Images/sort_menu.jpg")
        self.square = cv2.imread("Images/square.jpg")
        self.found = True

    def __del__(self):
        del self.gcvdata
        del self.overworld_template
        del self.item_full_template
        del self.item_menu_template
        del self.healing_rain
        del self.sort_menu
        del self.square
    
    def process(self, frame):
        self.gcvdata[0] = False
        frame = cv2.rectangle(frame, (1500,80), (1832,287), (255, 0, 0), 2)

        if self.imageMatch(frame, self.overworld_template, 0.98) or self.imageMatch(frame, self.square, 0.97):
            self.gcvdata[0] = True

        # Passing bools to GPC script to take action on.
        self.gcvdata[1] = self.imageMatch(frame, self.item_full_template, 0.98)
        self.gcvdata[2] = self.imageMatch(frame, self.item_menu_template, 0.98)
        self.gcvdata[3] = self.imageMatch(frame, self.healing_rain, 0.98)
        self.gcvdata[4] = self.imageMatch(frame, self.sort_menu, 0.98)

        return frame, self.gcvdata
    
    def imageMatch(self, frame, template, confidenceThreshold):
        similar = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(similar)

        # print('Confidence: %s' % confidence)
        if confidence > confidenceThreshold:
            return True
        
        return False
            
