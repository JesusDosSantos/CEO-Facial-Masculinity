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

padding_folder = 'C:/Users/JEDP1/Downloads/archive (4)/img_align_celeba/img_align_celeba'
padding_output_folder = "CelebA_aligned"
os.makedirs(padding_output_folder, exist_ok=True)

male_output_folder = "celeb_male_padded"
os.makedirs(male_output_folder, exist_ok=True)

female_output_folder = "celeb_female_padded"
os.makedirs(female_output_folder, exist_ok=True)

#for other datasets
'''
padding_folder = "train"
padding_output_folder = "train_padded"
os.makedirs(padding_output_folder, exist_ok=True)

male_output_folder = "male_padded"
os.makedirs(male_output_folder, exist_ok=True)

female_output_folder = "female_padded"
os.makedirs(female_output_folder, exist_ok=True)
'''

input_folder = "test images" #change this to the folder with the images
output_folder = "images_with_faces NEW2" #change this to the folder where you want the images with faces to be saved
RESOLUTION_THRESHOLD = (150, 150) # Minimum resolution for the cropped face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def face_orientation_score(face_image):
    img = np.array(face_image)
    img = img[:, :, ::-1].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        return 10000000

    landmarks = predictor(gray, faces[0])

    nose_center = np.array([landmarks.part(30).x, landmarks.part(30).y])
    left_face_outer = np.array([landmarks.part(0).x, landmarks.part(0).y])
    right_face_outer = np.array([landmarks.part(16).x, landmarks.part(16).y])
    chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
    left_eye_center = np.array([(landmarks.part(37).x + landmarks.part(40).x) // 2,
                                (landmarks.part(37).y + landmarks.part(40).y) // 2])
    right_eye_center = np.array([(landmarks.part(43).x + landmarks.part(46).x) // 2,
                                 (landmarks.part(43).y + landmarks.part(46).y) // 2])
    eyes_midpoint = (left_eye_center + right_eye_center) / 2

    left_distance = np.linalg.norm(nose_center - left_face_outer)
    right_distance = np.linalg.norm(nose_center - right_face_outer)
    tilt_score = abs(left_distance / right_distance - 1)

    eye_nose_distance = np.linalg.norm(nose_center - eyes_midpoint)
    nose_chin_distance = np.linalg.norm(nose_center - chin)
    up_down_score = abs(eye_nose_distance / nose_chin_distance - 1)

    combined_score = 0.7 * tilt_score + 0.3 * up_down_score

    return combined_score

def extended_crop(image, top, right, bottom, left, padding=30):
    height, width, _ = image.shape
    top = max(top - padding, 0)
    left = max(left - padding, 0)
    bottom = min(bottom + padding, height)
    right = min(right + padding, width)

    return Image.fromarray(image[top:bottom, left:right])


def load_image_safe(path):
    try:
        return face_recognition.load_image_file(path)
    except Exception as e:
        return None

def align_face(face_image):
    numpy_image = np.array(face_image)

    # Get face landmarks
    face_landmarks_list = face_recognition.face_landmarks(numpy_image)

    if not face_landmarks_list:
        return face_image

    # Calculate the centers of the eyes
    left_eye = np.array(face_landmarks_list[0]['left_eye'])
    right_eye = np.array(face_landmarks_list[0]['right_eye'])
    left_eye_center = left_eye.mean(axis=0)
    right_eye_center = right_eye.mean(axis=0)

    # Compute the angle between the eye centers
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Compute the center for the rotation
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) * 0.5, (left_eye_center[1] + right_eye_center[1]) * 0.5)

    # Adjust the rotation matrix to prevent cropping
    abs_cos = abs(np.cos(angle))
    abs_sin = abs(np.sin(angle))
    height, width = numpy_image.shape[:2]
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation center based on the new dimensions
    rotation_matrix = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1)
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Apply the rotation
    rotated_image = cv2.warpAffine(numpy_image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    return Image.fromarray(rotated_image)


def Choose_Best():
    for i, img in tqdm(enumerate(os.listdir(padding_folder))):
        paddingImg_path = os.path.join(padding_folder, img)
        image = load_image_safe(paddingImg_path)
        if image is None:
            continue

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) != 1:
            continue

        top, right, bottom, left = face_locations[0]
        face_image = extended_crop(image, top, right, bottom, left)
        face_image = align_face(face_image)

        # Check image resolution
        width, height = face_image.size
        if width < RESOLUTION_THRESHOLD[0] or height < RESOLUTION_THRESHOLD[1]:
            continue

        score = face_orientation_score(face_image)
        if score < 0.35:
            #save image in new folder "train clean" without using shutil
            face_image.save(os.path.join(padding_output_folder, img))

def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def predict_gender_image(input_path: str):
    img = cv2.imread(input_path)
    faces = get_faces(img)
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = img[start_y: end_y, start_x: end_x]

        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence_score = gender_preds[0][0]

        return gender_confidence_score

def divide_genders():
    for i, img in tqdm(enumerate(os.listdir(padding_output_folder))):
        paddingImg_path = os.path.join(padding_output_folder, img)
        image = load_image_safe(paddingImg_path)
        if image is None:
            continue
        else:
            image = align_face(image)

            if isinstance(image, np.ndarray):
                continue

            gender_confidence_score = predict_gender_image(paddingImg_path)

            if gender_confidence_score is None:
                continue

            if gender_confidence_score < 0.5:
                gender = "female"
                image.save(os.path.join(female_output_folder, img))
            else:
                gender = "male"
                image.save(os.path.join(male_output_folder, img))

def calculate_averageness(landmarks, average_landmarks):
    # Calculate the absolute difference between landmarks and average landmarks
    deviation = np.abs(landmarks - average_landmarks)
    # Take the mean to get a single averageness score
    averageness_score = np.mean(deviation)
    return averageness_score


def normalize_landmarks(landmarks, bounding_box):
    # Normalize the landmarks based on the bounding box of the detected face
    landmarks_normalized = [(x - bounding_box.left(), y - bounding_box.top()) for (x, y) in landmarks]
    width = bounding_box.width()
    height = bounding_box.height()
    landmarks_normalized = [(x / width, y / height) for (x, y) in landmarks_normalized]
    return landmarks_normalized

def get_celeb_male_averageness():
    measurements = []
    all_landmarks = []

    for i, img in tqdm(enumerate(os.listdir(male_output_folder))):
        image_path = os.path.join(male_output_folder, img)
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

    average_landmarks = np.mean(all_landmarks, axis=0)
    np.save('male_celeb_avg_landmarks.npy', average_landmarks)
    return average_landmarks

def get_celeb_female_averageness():
    measurements = []
    all_landmarks = []

    for i, img in tqdm(enumerate(os.listdir(female_output_folder))):
        image_path = os.path.join(female_output_folder, img)
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

    average_landmarks = np.mean(all_landmarks, axis=0)
    np.save('female_celeb_avg_landmarks.npy', average_landmarks)
    return average_landmarks

def get_female_features(average_landmarks):
    measurements = []
    for i, img in tqdm(enumerate(os.listdir(female_output_folder))):
        image_path = os.path.join(female_output_folder, img)
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

        averageness_score = calculate_averageness(landmarks_normalized, average_landmarks)
        measurements.append([img, averageness_score])
    df = pd.DataFrame(measurements, columns=["image", "averageness"])
    df.to_csv("female_celeb_averageness.csv", index=False)

def get_male_features(average_landmarks):
    measurements = []
    for i, img in tqdm(enumerate(os.listdir(male_output_folder))):
        image_path = os.path.join(male_output_folder, img)
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

        averageness_score = calculate_averageness(landmarks_normalized, average_landmarks)
        measurements.append([img, averageness_score])
    df = pd.DataFrame(measurements, columns=["image", "averageness"])
    df.to_csv("male_celeb_averageness.csv", index=False)

def predict_gender_name(id):
    id = id.split('-')[1]
    id = id.split('_')[0]
    id = int(id)
    name = df.loc[df['execid'] == id, 'exec_fname'].iloc[0]

    pred = name_detector.get_gender(name)

    if pred == "female":
        return 0
    elif pred == "male":
        return 1
    else:
        return 0.5

def get_genders(ImagePath, CEO_folder):
    image_pred = predict_gender_image(ImagePath)
    name_pred = predict_gender_name(CEO_folder)

    if image_pred is None:
        return None

    gender_confidence_score = 0.51 * image_pred + 0.49 * name_pred

    if gender_confidence_score < 0.5:
        gender = "female"
        return gender

    else:
        gender = "male"
        return gender

def get_celeb_averageness(male_averageness, female_averageness):
    measurements = []

    for CEO_folder in tqdm(os.listdir(output_folder)):
        image_folder_path = os.path.join(output_folder, CEO_folder, "Best")

        # Assuming there is only one image in the 'Best' folder.
        for image_name in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            if len(faces) == 0:
                continue

            landmarks = predictor(gray, faces[0])
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            landmarks_normalized = normalize_landmarks(landmarks_array, faces[0])

            gender = get_genders(image_path, CEO_folder)

            if gender == None:
                continue

            elif gender == 'female':
                averageness_score = calculate_averageness(landmarks_normalized, female_averageness)

            else:
                averageness_score = calculate_averageness(landmarks_normalized, male_averageness)

            measurements.append([CEO_folder, averageness_score, gender])
            break

    df = pd.DataFrame(measurements, columns=["CEO", "Averageness", "Gender"])
    df.to_csv("sample_vs_celeb_averageness.csv", index=False)

def get_sample_averageness():
    measurements = []
    female_landmarks = []
    male_landmarks = []

    for CEO_folder in tqdm(os.listdir(output_folder)):
        image_folder_path = os.path.join(output_folder, CEO_folder, "Best")

        # Assuming there is only one image in the 'Best' folder.
        for image_name in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            if len(faces) == 0:
                continue

            landmarks = predictor(gray, faces[0])
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            landmarks_normalized = normalize_landmarks(landmarks_array, faces[0])

            gender = get_genders(image_path, CEO_folder)

            if gender == None:
                continue

            elif gender == 'female':
                female_landmarks.append(landmarks_normalized)

            else:
                male_landmarks.append(landmarks_normalized)

            break

    average_female_landmarks = np.mean(female_landmarks, axis=0)
    average_male_landmarks = np.mean(male_landmarks, axis=0)

    for CEO_folder in tqdm(os.listdir(output_folder)):
        image_folder_path = os.path.join(output_folder, CEO_folder, "Best")

        # Assuming there is only one image in the 'Best' folder.
        for image_name in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            if len(faces) == 0:
                continue

            landmarks = predictor(gray, faces[0])
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            landmarks_normalized = normalize_landmarks(landmarks_array, faces[0])

            gender = get_genders(image_path, CEO_folder)

            if gender == None:
                continue

            elif gender == 'female':
                averageness_score = calculate_averageness(landmarks_normalized, average_female_landmarks)

            else:
                averageness_score = calculate_averageness(landmarks_normalized, average_male_landmarks)

            measurements.append([CEO_folder, averageness_score, gender])
            break

        df = pd.DataFrame(measurements, columns=["CEO", "Averageness", "Gender"])
        df.to_csv("sample_averageness.csv", index=False)

def plot_average_faces(male_landmarks, female_landmarks):
    fig, ax = plt.subplots()
    ax.plot(-male_landmarks[:, 0], -male_landmarks[:, 1], 'b.-', label='Average Male Face')
    #ax.plot(-female_landmarks[:, 0], -female_landmarks[:, 1], 'r.-', label='Average Female Face')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.set_title('Average Faces')
    ax.axis('equal')
    plt.show()

def main():
    #Assume Executives folders have the best image of each CEO

    #Choose_Best() is run on the raw celeb folder to create a new folder only with celebs looking towards the camera
    Choose_Best()

    #divide_genders() creates two new folders of celebrities, separated by gender
    divide_genders()

    #the following two fucntions get the average vector for male and female celebrities respectively
    celeb_male_averageness = get_celeb_male_averageness()
    celeb_female_averageness = get_celeb_female_averageness()

    #the following two functions create a csv file with the averageness score for for each celebrity, to test if the model is working
    #they do not have to execute for everything to work
    '''
    get_male_features(celeb_male_averageness)
    get_female_features(celeb_male_averageness)
    '''

    plot_average_faces(celeb_male_averageness, celeb_female_averageness)

    #Finally, this function creates a csv file with the averageness score for each CEO against the celebrity average vector
    get_celeb_averageness(celeb_male_averageness, celeb_female_averageness)

if __name__ == "__main__":
    main()