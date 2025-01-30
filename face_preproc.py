import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image


def main():
    data_dir = './data'

    file = os.listdir(data_dir)
    people = [f.replace(".MOV", "") for f in file if ".MOV" in f]


    for name in people:
        
        # Save directory for cropped faces
        save_dir = f'{data_dir}/faces/{name}'
        os.makedirs(save_dir, exist_ok=True)

        # Create face detector
        mtcnn = MTCNN(keep_all=True)

        # Load a single image and display
        for j in range(len(os.listdir(f'{data_dir}/ready/{name}'))):
            v_cap = cv2.VideoCapture(f'{data_dir}/ready/{name}/{j}.jpg')
            success, frame = v_cap.read()
            frame = Image.fromarray(frame)

            # Detect face
            boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

            try:
                for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                    # Crop the face and save
                    cropped_face = frame.crop((box[0], box[1], box[2], box[3]))
                    cropped_face.save(os.path.join(save_dir, f'face_{j}.jpg'))
            except:
                pass

main()