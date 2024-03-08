import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
import torch
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load pre-stored embeddings
file_path = 'C:/coding/projects/parking/mediapipe-rec/mp_pre_stored_embeddings.pt'
pre_stored_embeddings = torch.load(file_path)

# Set the threshold
threshold = 0.07


# Function to calculate L1 distance between two embeddings
def calculate_l1_distance(embedding1, embedding2):
    return torch.nn.functional.l1_loss(embedding1, embedding2).item()

# Function to find the index of the closest matching image and calculate distance
def find_closest_image(embedding):
    min_distance = float('inf')
    nearest_img = None
    for stored_img, stored_embedding in pre_stored_embeddings.items():
        distance = calculate_l1_distance(embedding, stored_embedding)
        if distance < min_distance:
            min_distance = distance
            nearest_img = stored_img
    return nearest_img, min_distance

# Start capturing video from the webcam.

cap = cv2.VideoCapture(0)


# Initialize a dictionary to store queues for each face detected
face_queues = {}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            x_margin = 20
            y_margin = 20
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x -= x_margin
            y -= y_margin
            w += 2 * x_margin
            h += 2 * y_margin
            x, y, w, h = max(x, 0), max(y, 0), min(w, iw-x), min(h, ih-y)

            if w * h == 0:
                continue  # Skip this detection and continue with the next

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = image[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (160, 160))
            face_tensor = torch.from_numpy(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            img_embedding = resnet(face_tensor.unsqueeze(0))

            nearest_img, distance = find_closest_image(img_embedding)
            confidence_score = max(0, (distance / threshold) * 100)

            if distance < threshold:
                identification_text = nearest_img
                confidence_score = max(0, (distance / threshold) * 100)
            else:
                identification_text = 'Unknown'

            # Check if face ID exists in the dictionary, if not, create a new queue
            if identification_text not in face_queues:
                face_queues[identification_text] = deque(maxlen=10)

            # Add the identification_text to the respective face queue
            face_queues[identification_text].append(identification_text)

            # Check if the last 10 frames for this face have the same identification_text
            if len(set(face_queues[identification_text])) == 1:
                identification_text = f'{identification_text} ({confidence_score})'
                # Position the text above the bounding box
                cv2.putText(image, identification_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
