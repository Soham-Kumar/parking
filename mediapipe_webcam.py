import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
import torch


# Initialize
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load pre-stored embeddings
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



# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

nearest_img = None
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find faces.
    results = face_detection.process(image)

    # Convert the image color back so it can be displayed.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the face.
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
        

            # Draw the bounding box around the face.
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Annotate the coordinates on the image.
            coordinate_text = f"X: {x}, Y: {y}, W: {w}, H: {h}"
            cv2.putText(image, coordinate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1)
            
            # Extract the face ROI and resize it to 96x96.
            face_roi = image[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (160, 160))
            face_tensor = torch.from_numpy(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            img_embedding = resnet(face_tensor.unsqueeze(0))
            nearest_img = find_closest_image(img_embedding)
            cv2.putText(image, f'Index: {nearest_img}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    

    # Display the annotated image.
    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
