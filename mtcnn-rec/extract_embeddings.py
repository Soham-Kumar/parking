import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os


mtcnn = MTCNN(image_size=160, margin=10)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

pre_stored_embeddings = {}


folder_path = "embeddings"

# List all files in the directory
files = os.listdir(folder_path)

# Filter out non-image files
image_files = [file for file in files if file.endswith(('.jpg'))]
print("Image files:", image_files)

# Process each image file
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    img = cv2.imread(image_path)
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    pre_stored_embeddings[image_file] = img_embedding


print(pre_stored_embeddings)

file_path = r'C:\coding\projects\parking\MTCNN\mtcnn_pre_stored_embeddings.pt'
torch.save(pre_stored_embeddings, file_path)
