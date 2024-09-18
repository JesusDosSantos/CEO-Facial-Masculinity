import os
import cv2
import numpy as np
import torch
import face_recognition
import math
from math import atan2, pi
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from hopenet import Hopenet
from sklearn.cluster import DBSCAN
import dlib
import time
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

input_folder = "test images" #change this to the folder with the images
output_folder = "images_with_faces NEW" #change this to the folder where you want the images with faces to be saved
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
        print(f"Error loading image {path}: {e}")
        return None


def pool_faces_in_folder(image_folder_path): #try using weights
    encodings = []
    image_paths = []

    for image_name in sorted(os.listdir(image_folder_path)):
        image_path = os.path.join(image_folder_path, image_name)
        image = load_image_safe(image_path)
        if image is None:
            continue

        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            continue

        encoding = face_recognition.face_encodings(image, face_locations)
        if len(encoding) == 0:
            continue

        encodings.append(encoding[0])
        image_paths.append(image_path)

    if not encodings:
        return []

    face_pools = []

    for i in range(len(encodings)):
        pool = [image_paths[i]]
        for j in range(i + 1, len(encodings)):
            if face_recognition.compare_faces([encodings[i]], encodings[j])[0]:
                pool.append(image_paths[j])
        face_pools.append(pool)

    face_pools.sort(key=lambda x: (-len(x), image_paths.index(x[0])))

    return face_pools[0] if face_pools and len(face_pools[0]) > 1 else []

def is_grayscale(img):
    return np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2])


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


#IMPORTANT FUNCTION FOR IMAGE PROCESSING, contains functions above
def Choose_Best():
    for CEO_folder in tqdm(os.listdir(input_folder)):
        image_folder_path = os.path.join(input_folder, CEO_folder)

        largest_face_pool = pool_faces_in_folder(image_folder_path)
        if not largest_face_pool:
            print(f"No valid face pool for {CEO_folder}. Skipping...")
            continue

        images_and_scores = []

        for image_path in largest_face_pool:
            image = load_image_safe(image_path)
            if image is None:
                continue

            #separate grey scale pictures, for old pictures
            #if is_grayscale(image):
                #continue

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
            images_and_scores.append((face_image, score))

        if not images_and_scores:
            print(f"No valid images to save for {CEO_folder}. Skipping...")
            continue

        images_and_scores.sort(key=lambda x: x[1])

        best_path = os.path.join(output_folder, CEO_folder, "Best")
        os.makedirs(best_path, exist_ok=True)
        align_face(images_and_scores[0][0]).save(os.path.join(best_path, "best_image.jpg"))

        cropped_path = os.path.join(output_folder, CEO_folder, "Cropped")
        os.makedirs(cropped_path, exist_ok=True)
        for idx, (img, _) in enumerate(images_and_scores):
            img.save(os.path.join(cropped_path, f"cropped_{idx}.jpg"))


def get_face_points(points, method='average', top='eyebrow'):
    width_left, width_right = points[0], points[16]

    if top == 'eyebrow':
        top_left = points[18]
        top_right = points[25]

    elif top == 'eyelid':
        top_left = points[37]
        top_right = points[43]

    else:
        raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')

    bottom_left, bottom_right = points[50], points[52]

    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])

    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])

    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)

    ## Move the line just a little above the top of the eye to the eyelid
    if top == 'eyelid':
        coords = (coords[0], coords[1], coords[2] - 4, coords[3])

    return {'top_left': (coords[0], coords[2]),
            'bottom_left': (coords[0], coords[3]),
            'top_right': (coords[1], coords[2]),
            'bottom_right': (coords[1], coords[3])
            }

def FWHR_calc(corners):
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(width) / float(height)


def show_box(image, corners):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size

    ## Automatically determine width of the line depending on size of picture
    line_width = math.ceil(h / 100)

    d = ImageDraw.Draw(pil_image)
    d.line([corners['bottom_left'], corners['top_left']], width=line_width)
    d.line([corners['bottom_left'], corners['bottom_right']], width=line_width)
    d.line([corners['top_left'], corners['top_right']], width=line_width)
    d.line([corners['top_right'], corners['bottom_right']], width=line_width)

    return pil_image

def load_image(path):
    return face_recognition.load_image_file(path)

def get_fwhr(image_path, method='average', top='eyelid'):
    image = load_image(image_path)
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]

    corners = get_face_points(landmarks_as_tuples, method=method, top=top)
    fwh_ratio = FWHR_calc(corners)

    box_image = show_box(image, corners)

    return fwh_ratio, box_image

def get_Nasal_Index(landmarks):
    nose_width = np.linalg.norm(np.array([landmarks.part(35).x, landmarks.part(35).y]) -
                                np.array([landmarks.part(31).x, landmarks.part(31).y]))
    nose_height = landmarks.part(33).y - landmarks.part(27).y
    nasal_index = nose_width / nose_height
    return nasal_index

def get_Lip_Fullness(landmarks):
    # Calculating heights for upper and lower lips using three measurements
    # Upper lip fullness
    top_of_upper_lip = np.array([landmarks.part(51).x, landmarks.part(51).y])  # Top of the upper lip (center point)
    center_of_lips = np.array([landmarks.part(62).x, landmarks.part(62).y])  # Center point where lips meet
    upper_lip_fullness = np.linalg.norm(top_of_upper_lip - center_of_lips)

    # Lower lip fullness
    top_of_lower_lip = np.array([landmarks.part(66).x, landmarks.part(66).y])  # Center of the top of the lower lip
    bottom_of_lower_lip = np.array([landmarks.part(57).x, landmarks.part(57).y])  # Bottom of the lower lip (center point)
    lower_lip_fullness = np.linalg.norm(top_of_lower_lip - bottom_of_lower_lip)

    # Computing lip fullness ratio
    lip_fullness_ratio = upper_lip_fullness / lower_lip_fullness

    return lip_fullness_ratio

def get_Jawline_Angle(landmarks):
    # Extracting the three landmarks
    chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
    jaw_left = np.array([landmarks.part(4).x, landmarks.part(4).y])
    jaw_right = np.array([landmarks.part(12).x, landmarks.part(12).y])

    # Computing the vectors
    vector_1 = chin - jaw_left
    vector_2 = chin - jaw_right

    # Computing the cosine of the angle using the dot product
    cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

    # Computing the angle in degrees
    angle = np.degrees(np.arccos(clamp(cosine_angle, -1.0, 1.0)))

    return angle

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def calculate_angle(a, b, c):
    # Calculate the lengths of the triangle sides
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ac = np.linalg.norm(a - c)

    # Apply the Law of Cosines to find the angle at point b
    cosine_angle = (ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def eyebrow_angle(landmarks):
    left_start = np.array([landmarks.part(17).x, landmarks.part(17).y])
    left_peak = np.array([landmarks.part(19).x, landmarks.part(19).y])
    left_end = np.array([landmarks.part(21).x, landmarks.part(21).y])

    right_start = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_peak = np.array([landmarks.part(24).x, landmarks.part(24).y])
    right_end = np.array([landmarks.part(26).x, landmarks.part(26).y])

    left_angle = calculate_angle(left_start, left_peak, left_end)
    right_angle = calculate_angle(right_start, right_peak, right_end)

    return (left_angle + right_angle) / 2


def face_roundness(landmarks):
    width = np.linalg.norm(
        np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
    height = landmarks.part(8).y - landmarks.part(19).y
    return width / height


def chin_angle(landmarks):
    left_point = np.array([landmarks.part(5).x, landmarks.part(5).y])
    vertex = np.array([landmarks.part(8).x, landmarks.part(8).y])
    right_point = np.array([landmarks.part(11).x, landmarks.part(11).y])

    return calculate_angle(left_point, vertex, right_point)


def philtrum(landmarks):
    nose_to_lip = np.linalg.norm(
        np.array([landmarks.part(33).x, landmarks.part(33).y]) - np.array([landmarks.part(51).x, landmarks.part(51).y]))
    upper_face_length = landmarks.part(27).y - landmarks.part(19).y
    return nose_to_lip / upper_face_length


def lip_width_to_face_width(landmarks):
    lip_width = np.linalg.norm(
        np.array([landmarks.part(48).x, landmarks.part(48).y]) - np.array([landmarks.part(54).x, landmarks.part(54).y]))
    face_width = np.linalg.norm(
        np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
    return lip_width / face_width


def lip_fullness_to_face_width(landmarks):
    # Upper lip fullness
    top_of_upper_lip = np.array([landmarks.part(51).x, landmarks.part(51).y])  # Top of the upper lip (center point)
    center_of_lips = np.array([landmarks.part(62).x, landmarks.part(62).y])  # Center point where lips meet
    upper_lip_fullness = np.linalg.norm(top_of_upper_lip - center_of_lips)

    # Lower lip fullness
    top_of_lower_lip = np.array([landmarks.part(66).x, landmarks.part(66).y])  # Center of the top of the lower lip
    bottom_of_lower_lip = np.array([landmarks.part(57).x, landmarks.part(57).y])  # Bottom of the lower lip (center point)
    lower_lip_fullness = np.linalg.norm(top_of_lower_lip - bottom_of_lower_lip)

    # Total lip fullness
    total_lip_fullness = upper_lip_fullness + lower_lip_fullness

    # Facial width
    face_width = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))

    return total_lip_fullness / face_width

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

def get_features():
    measurements = []
    all_landmarks = []

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
            all_landmarks.append(landmarks_normalized)
            break

    average_landmarks = np.mean(all_landmarks, axis=0)

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

            averageness_score = calculate_averageness(landmarks_normalized, average_landmarks)

            FWHr, box_image = get_fwhr(image_path, method='average', top='eyelid')

            Nasal_Index = get_Nasal_Index(landmarks)

            Lip_Fullness = get_Lip_Fullness(landmarks)

            Jawline_Angle = get_Jawline_Angle(landmarks)

            EYEBROW = eyebrow_angle(landmarks)

            FACE_SHAPE = face_roundness(landmarks)

            CHIN_ANGLE = chin_angle(landmarks)

            PHILTRUM = philtrum(landmarks)

            Lip_to_Face_Width = lip_width_to_face_width(landmarks)

            Lip_Fullness_to_Face_Width = lip_fullness_to_face_width(landmarks)

            box_image = cv2.cvtColor(np.array(box_image), cv2.COLOR_RGB2BGR)
            #save image with box
            cv2.imwrite(os.path.join(output_folder, CEO_folder, "Best", "box_image.jpg"), box_image)

            measurements.append([CEO_folder, FWHr, Nasal_Index, Lip_Fullness, Jawline_Angle,
                   EYEBROW, FACE_SHAPE, CHIN_ANGLE, PHILTRUM,
                   Lip_to_Face_Width, Lip_Fullness_to_Face_Width, averageness_score])
            break
    # Saving measurements to CSV using pandas


    df = pd.DataFrame(measurements, columns=["ID", "FWHr", "Nasal Index", "Lip Fullness", "Jawline Angle",
                'EYEBROW', 'FACE_SHAPE', 'CHIN_ANGLE',
                'PHILTRUM', 'Lip_to_Face_Width', 'Lip_Fullness_to_Face_Width','averageness_score'])
    df.to_csv('facial_features8.csv', index=False)

    return



def main():
    Choose_Best()
    get_features()

if __name__ == "__main__":
    main()
