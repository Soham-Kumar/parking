import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os


mtcnn = MTCNN(image_size=160, margin=10)
# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
# )


resnet = InceptionResnetV1(pretrained='vggface2').eval()

file_path = 'pre_stored_embeddings.pt'
pre_stored_embeddings = torch.load(file_path)


# Function to calculate L1 distance between two embeddings
def calculate_l1_distance(embedding1, embedding2):
    return torch.nn.functional.l1_loss(embedding1, embedding2).item()


# Function to find the index of closest matching image
def find_closest_image(embedding):
    min_distance = float('inf')
    nearest_img = None
    for stored_img, stored_embedding in pre_stored_embeddings.items():
        distance = calculate_l1_distance(embedding, stored_embedding)
        if distance < min_distance:
            min_distance = distance
            nearest_img = stored_img
    return nearest_img



# Open webcam capture
cap = cv2.VideoCapture(0)

nearest_img = None
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Ensure coordinates are within frame dimensions
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            if x2 - x1 > 0 and y2 - y1 > 0:  # Check for positive width and height
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if frame_count % 5 == 0:
                    face_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if face_roi.size > 0:
                        resized_face = cv2.resize(face_roi, (96, 96))
                        face_tensor = torch.from_numpy(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                        img_embedding = resnet(face_tensor.unsqueeze(0))
                        nearest_img = find_closest_image(img_embedding)
                        cv2.putText(frame, f'Index: {nearest_img}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow('Webcam', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
