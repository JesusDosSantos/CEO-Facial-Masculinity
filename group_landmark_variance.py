import os, sys, csv, cv2, torch, face_recognition, math, dlib, time, re
import numpy as np
from math import atan2, pi
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from hopenet import Hopenet
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
import gender_guesser.detector as gender
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv("ceo_cfo.csv")

GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


name_detector = gender.Detector()

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

RESOLUTION_THRESHOLD = (150, 150) # Minimum resolution for the cropped face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def calculate_landmark_variance(IDs, folder):
    def load_image_safe(path):
        try:
            return face_recognition.load_image_file(path)
        except Exception as e:
            return None

    def normalize_landmarks(landmarks, bounding_box):
        # Normalize the landmarks based on the bounding box of the detected face
        landmarks_normalized = [(x - bounding_box.left(), y - bounding_box.top()) for (x, y) in landmarks]
        width = bounding_box.width()
        height = bounding_box.height()
        landmarks_normalized = [(x / width, y / height) for (x, y) in landmarks_normalized]
        return landmarks_normalized

    all_landmarks = []
    for ID in IDs:
        image_path = os.path.join(folder, ID)
        image = load_image_safe(image_path)
        if image is None:
            continue
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        if len(faces) == 0:
            continue

        landmarks = predictor(gray, faces[0])
        landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        landmarks_normalized = normalize_landmarks(landmarks_array, faces[0])
        all_landmarks.append(landmarks_normalized)
        break

    x_coords, y_coords = zip(*[coord for face in all_landmarks for coord in face])

    # Calculate the sample variance for x and y coordinates
    variance_x = np.var(x_coords, ddof=1)
    variance_y = np.var(y_coords, ddof=1)

    overall_variance = (variance_x + variance_y) / 2

    print(overall_variance)


def main():
    #Try with celebs first

    #males
    dfMaleCelebs = pd.read_csv('male_celeb_averageness.csv')
    dfMaleCelebsSorted = dfMaleCelebs.sort_values(['averageness'])

    #put in lists the first 10% ids, the 45% to 55%, and the 90% to 100%
    maleCelebsSortedIds = dfMaleCelebsSorted['image'].tolist()

    tenPercent = round(0.1*(len(maleCelebsSortedIds)))

    mostAttractiveMales = maleCelebsSortedIds[0:tenPercent]
    averageAttractiveMales = maleCelebsSortedIds[round(tenPercent*4.5):round(tenPercent*5.5)]
    leastAttractiveMales = maleCelebsSortedIds[-tenPercent:]

    print(len(leastAttractiveMales))
    print(maleCelebsSortedIds[tenPercent*9:tenPercent*10])
    '''
    calculate_landmark_variance(mostAttractiveMales, 'celeb_male_padded')
    calculate_landmark_variance(averageAttractiveMales, 'celeb_male_padded')
    '''
    calculate_landmark_variance(leastAttractiveMales, 'celeb_male_padded')
    calculate_landmark_variance(maleCelebsSortedIds[tenPercent*9+2:tenPercent*10+2], 'celeb_male_padded')




if __name__ == "__main__":
    main()

'''    
for i in range(10):
    males  = maleCelebsSortedIds[round(tenPercent*i):round(tenPercent*i+1)]
    variance = calculate_landmark_variance(males, 'celeb_male_padded')
    print('From ',i, 'to', i+1, ' ', variance) 0.06594614881109595
'''