import os
import cv2

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.gcvdata = bytearray([0x00, 0x00, 0x00, 0x00, 0x00])
        self.attack_combo = cv2.imread("Images/attack_combo_small.jpg")
        self.mimic_farm = cv2.imread("Images/just_valens.jpg")


    def __del__(self):
        del self.gcvdata
        del self.attack_combo
        del self.mimic_farm

    
    def process(self, frame):
        self.gcvdata[0] = False
        frame = cv2.rectangle(frame, (700,100), (1400,500), (255, 0, 0), 2)
        #picture = frame[700:100, 1200:500]
        #picture = frame[700:1200, 100:500]
        picture = frame[100:700, 500:1400]

        # Passing bools to GPC script to take action on.
        self.gcvdata[0] = self.imageMatch(picture, self.attack_combo, 0.97)
        
        # TODO: Figure out how to enable this via a trigger in the GPC script, otherwise will spend too much time reading this screen
        # And miss even more combos most likely. 
        # PS this does work! when uncommented and fighting mimic.
        #self.gcvdata[1] = self.imageMatch(frame, self.mimic_farm, 0.97)
        
        if self.gcvdata[1]:
            print('Found a fine wine on screen!')

        return frame, self.gcvdata
    
    def imageMatch(self, frame, template, confidenceThreshold):
        similar = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(similar)

        
        if confidence > confidenceThreshold:
            print('Confidence: %s' % confidence)
            return True
        
        return False
            
