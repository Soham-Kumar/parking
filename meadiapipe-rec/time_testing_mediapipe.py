import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import time

# Load pre-stored embeddings
file_path = r'C:\coding\projects\parking\mediapipe-rec\mp_pre_stored_embeddings.pt'
pre_stored_embeddings = torch.load(file_path)

# Instantiate ResNet
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to calculate embedding
def calculate_embedding(image_path):
    img = Image.open(image_path)
    img_cvt = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if not results.detections:
        return None
    
    detection = results.detections[0]
    ih, iw, _ = img_rgb.shape
    bboxC = detection.location_data.relative_bounding_box
    xmin, ymin, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(iw - 1, xmin + w), min(ih - 1, ymin + h)
    
    face_img = img.crop((xmin, ymin, xmax, ymax))
    face_img = face_img.resize((160, 160))
    face_tensor = F.to_tensor(face_img).unsqueeze(0)
    face_tensor = face_tensor * 2 - 1
    
    img_embedding = resnet(face_tensor)
    
    return img_embedding

# Function to calculate L1 distance
def calculate_l1_distance(embedding1, embedding2):
    return torch.nn.functional.l1_loss(embedding1, embedding2).item()

# Function to find the closest embedding
def find_closest_embedding(input_embedding, pre_stored_embeddings):
    min_distance = float('inf')
    closest_key = None

    for key, stored_embedding in pre_stored_embeddings.items():
        distance = calculate_l1_distance(input_embedding, stored_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_key = key

    return closest_key

img_path = r'C:\coding\projects\parking\image_dataset\image_one\Zelma_Novelo_0001.jpg'


# ------------------------------- Calculate Time for calculating embeddings -------------------------------
start_time_embedding = time.perf_counter()
input_embedding = calculate_embedding(img_path)
end_time_embedding = time.perf_counter()
print(f"Time to calculate new embedding: {end_time_embedding - start_time_embedding:0.4f} seconds")


if input_embedding is not None:
    start_time_calc = time.perf_counter()
    closest_key = find_closest_embedding(input_embedding, pre_stored_embeddings)
    end_time_calc = time.perf_counter()
    print(f"Time to match nearest embedding from 1000 embeddings: {end_time_calc - start_time_calc:0.4f} seconds")
else:
    print("No face detected.")