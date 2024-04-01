import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from facenet_pytorch import InceptionResnetV1


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load pre-stored embeddings
file_path = r'C:\coding\projects\parking\mediapipe-rec\small_pre_stored_embeddings.pt'
pre_stored_embeddings = torch.load(file_path)

# Variables
threshold = 0.035
angle = 30
queue_length = 600

# Initialize variables for identification
identification_text = 'Unknown'
confidence_score = "0.0"

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
    

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize a dictionary to store queues for each face detected
face_queues = {}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results_detection = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results_detection.detections:
        for detection in results_detection.detections:
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

            # Draw the bounding box around the detected face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = image[y:y+h, x:x+w]

            results_mesh = face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            img_h, img_w, img_c = face_roi.shape
            face_3d, face_2d = [], []

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    a, b, c = angles[0] * 360, angles[1] * 360, angles[2] * 360

                    if not (-int(angle) < a < int(angle) and -int(angle) < b < int(angle)):
                        continue

                    resized_face = cv2.resize(face_roi, (160, 160))
                    face_tensor = torch.from_numpy(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                    img_embedding = resnet(face_tensor.unsqueeze(0))

                    nearest_img, distance = find_closest_image(img_embedding)

                    if distance < threshold:
                        identification_text = nearest_img
                        confidence_score = max(0, (distance / threshold) * 100)
                    else:
                        identification_text = 'Unknown'

                    if identification_text not in face_queues:
                        face_queues[identification_text] = deque(maxlen=queue_length)

                    face_queues[identification_text].append(identification_text)

                    if len(set(face_queues[identification_text])) == 1:
                        identification_text = f'{identification_text} ({confidence_score})'
                        cv2.putText(image, identification_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
