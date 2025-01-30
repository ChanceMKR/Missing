# app.py
import streamlit as st
from PIL import Image
import face_check  # face_check.py 파일이 같은 디렉토리에 있어야 함

# 모델과 관련된 컴포넌트를 캐싱하여 한 번만 로드되도록 함
@st.cache_resource
def load_face_model():
    """
    face_check 모듈을 사용하여 모델과 관련된 컴포넌트를 로드.

    Returns:
        tuple: (model, transform, class_names, device)
    """
    model, transform, class_names, device = face_check.load_model()
    return model, transform, class_names, device

# 모델 로드
model, transform, class_names, device = load_face_model()

# Streamlit 애플리케이션 제목
st.title('얼굴 인식')

# 파일 업로더 생성 (이미지 파일만 허용)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # PIL을 사용하여 이미지 열기 및 RGB 변환
        image = Image.open(uploaded_file).convert("RGB")

        # 업로드한 이미지 표시
        st.image(image, caption="업로드한 이미지", use_container_width=True)

        # 예측 실행
        pred_class = face_check.predict_image(model, image, transform, class_names, device)

        # 예측 결과 출력
        st.write(f"**예측된 클래스:** {pred_class}")

    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")