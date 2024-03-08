import os
from PIL import Image
import numpy as np
import torch
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms.functional as F

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to embeddings folder
embeddings_folder = r"C:\coding\projects\parking\small_image_dataset"

pre_stored_embeddings = {}

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Create an inception resnet (in eval mode) and move it to GPU if available:
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to calculate embedding
def calculate_embedding(image_path, face_detection_module):
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')
    img_np = np.array(img_rgb)
    
    results = face_detection_module.process(img_np)
    if not results.detections:
        return None
    
    detection = results.detections[0]
    
    # Extract face coordinates
    ih, iw, _ = img_np.shape
    ymin, xmin, h, w = int(detection.location_data.relative_bounding_box.ymin * ih), \
                       int(detection.location_data.relative_bounding_box.xmin * iw), \
                       int(detection.location_data.relative_bounding_box.height * ih), \
                       int(detection.location_data.relative_bounding_box.width * iw)
    
    face_img = img.crop((xmin, ymin, xmin + w, ymin + h))
    face_img = face_img.resize((160, 160))
    face_tensor = F.to_tensor(face_img).unsqueeze(0).to(device)
    face_tensor = face_tensor * 2 - 1
    
    img_embedding = resnet(face_tensor)
    
    return img_embedding.cpu().detach()

count = 0

for filename in os.listdir(embeddings_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(embeddings_folder, filename)
        img_embedding = calculate_embedding(image_path, mp_face_detection)
        if img_embedding is not None:
            pre_stored_embeddings[filename] = img_embedding
            count += 1
            if count % 100 == 0:
                print("Processed", count, "images")

print(len(pre_stored_embeddings), "embeddings calculated")

# Save the embeddings to a file
file_path = r'C:\coding\projects\parking\mediapipe-rec\small_pre_stored_embeddings.pt'
torch.save(pre_stored_embeddings, file_path)
print("Embeddings saved to", file_path)
