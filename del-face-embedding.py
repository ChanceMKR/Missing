import os
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from tqdm import tqdm

def main():
    # 데이터 경로 설정
    data_dir = "data/ready"
    output_embeddings = {}

    # ArcFace 모델 로드
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    # 각 폴더(개인별)에서 이미지 불러오기
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            
            for img_name in tqdm(os.listdir(person_path), desc=f"Processing {person_name}"):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                faces = app.get(img)
                if faces:
                    embedding = faces[0].normed_embedding  # 얼굴 임베딩 추출
                    embeddings.append(embedding)
            
            if embeddings:
                output_embeddings[person_name] = np.mean(embeddings, axis=0)  # 평균 임베딩 저장

    # 임베딩 저장
    np.save("face_embeddings.npy", output_embeddings)

main()