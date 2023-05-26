import cv2
import mediapipe as mp

# class creation
class HandDetectorHelper():
    '''class HandDetectorHelper'''
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8,modelComplexity=0,trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 20 landmark points
        self.mpDrawingStyles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img) # process the frame
    #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    #Draw dots and connect them
                    self.mpDraw.draw_landmarks( \
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyles.get_default_hand_landmarks_style(),
                        self.mpDrawingStyles.get_default_hand_connections_style())
        return img
    
    def findPointsHands(self, img, draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Coordenadas da mão detectada
                coordinates_x_min, coordinates_x_max = int(min([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1]), \
                            int(max([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1])
                coordinates_y_min, coordinates_y_max = int(min([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0]), \
                            int(max([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0])
            return self.results.multi_hand_landmarks, coordinates_x_min, coordinates_y_min, coordinates_x_max, coordinates_y_max
        return 0, False, False, False, False
    
    def markPointsHands(self, img, draw=True, points=None):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame

        if points != None:
            for handLms in points:
                if draw:
                    self.mpDraw.draw_landmarks( \
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyles.get_default_hand_landmarks_style(),
                        self.mpDrawingStyles.get_default_hand_connections_style())
        return img
    
    def markRoiHands(self, img, coordinates_x_min, coordinates_y_min, coordinates_x_max, coordinates_y_max):
        cv2.rectangle(img, (coordinates_x_min-50, coordinates_y_min-50), (coordinates_x_max+50, coordinates_y_max+50), (0, 255, 0), 2)
        return img
            
    def findPosition(self,img, handNo=0, draw=True):
        """Lists the position/type of landmarks
        we give in the list and in the list ww have stored
        type and position of the landmarks.
        List has all the lm position"""

        lmlist = []

        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            #Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h,w,c = img.shape
                #find the position
                cx,cy = int(lm.x*w), int(lm.y*h) #center
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])

                # Draw circle for 0th landmark
                # Sobreescreve as juntas com um circulo vermelho (0,0,255)
                # if draw:
                #     cv2.circle(img,(cx,cy), 5 , (0,0,255), cv2.FILLED)

        return lmlist
