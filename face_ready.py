import os
import cv2

def main():
    data_dir = './data'

    file = os.listdir(data_dir)
    people = [f.replace(".MOV", "") for f in file if ".MOV" in f]

    for name in people:
        ready_path = f"{data_dir}/ready/{name}"

        # ready 디렉토리 생성   
        if not os.path.exists(ready_path):
            os.makedirs(ready_path)

        filepath = os.path.join(data_dir, f"{name}.MOV")
        vidcap = cv2.VideoCapture(filepath)

        # 동영상의 메타데이터에서 회전 정보 읽기
        rotate_code = None
        if vidcap.get(cv2.CAP_PROP_ORIENTATION_META) == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif vidcap.get(cv2.CAP_PROP_ORIENTATION_META) == 180:
            rotate_code = cv2.ROTATE_180
        elif vidcap.get(cv2.CAP_PROP_ORIENTATION_META) == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        count = 0
        while vidcap.isOpened():
            ret, image = vidcap.read()
            if not ret:
                break

            # 회전 정보가 있으면 이미지를 회전
            if rotate_code is not None:
                image = cv2.rotate(image, rotate_code)

            # 이미지 저장
            save_path = os.path.join(ready_path, f"{count}.jpg")
            cv2.imwrite(save_path, image)

            if count == 900:  # 최대 900개의 프레임만 저장
                break
            count += 1

        vidcap.release()

main()
