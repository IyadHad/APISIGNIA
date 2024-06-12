import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
#from tensorflow.keras.models import load_model
from flask import Flask,jsonify,request
from flask_restful import Resource, Api

app = Flask(__name__)
#api = Api(app)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
model = tf.keras.models.load_model('9mots500.h5')

class prediction(Resource):
    def __init__(self):
        self.actions = np.array(['Bonjour','Bravo','Ca va','Non','Oui','Au revoir','Pardon','Stp','Bien','Pas bien'])
        self.sequence = []
        self.sentence = []
        self.threshold = 0.8

    def draw_styled_landmarks(image, results):
        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        # Draw left hand landmarks
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        # Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self,results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def post(self):
        file = request.files['file']
        cap = cv2.VideoCapture(file)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                self.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
                #sequence.insert(0,keypoints)
                #sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.actions[np.argmax(res)])
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > self.threshold: 
                        if len(sentence) > 0: 
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence=(self.actions[np.argmax(res)])
                        else:
                            sentence=(self.actions[np.argmax(res)])

                    # if len(sentence) > 5: 
                    #     sentence = sentence[-5:]

                    # Viz probabilities
                    #image = prob_viz(res, actions, image, colors)
            return jsonify({"signe" : sentence})

#api.add_resource(prediction,'/predict',methods=['POST'])

app.route('/prediction',methods=['POST'])
def resultat():
    var = prediction()
    return var.post()

if __name__ == '__main__':
    app.run(debug=False)
    
