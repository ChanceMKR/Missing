import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image


# ArcFace 백본 임포트
sys.path.append('insightface/recognition/arcface_torch/backbones')
from iresnet import iresnet50  # 필요에 따라 iresnet100 등으로 변경 가능

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_model(backbone_path='weights/backbone.pth', finetuned_model_path='best_arcface_finetune.pt', data_dir='data/faces'):
    """
    모델과 관련된 모든 컴포넌트를 로드하는 함수.

    Args:
        backbone_path (str): 사전 학습된 ArcFace 백본의 경로.
        finetuned_model_path (str): 파인튜닝된 모델 가중치의 경로.
        data_dir (str): 학습된 데이터셋의 디렉토리 경로.

    Returns:
        model (nn.Module): 로드된 모델.
        transform (transforms.Compose): 이미지 변환 파이프라인.
        class_names (list): 클래스 이름 리스트.
        device (torch.device): 사용 중인 디바이스.
    """
    # 백본 모델 로드
    arcface_backbone = iresnet50(pretrained=False, fp16=False)
    state_dict = torch.load(backbone_path, map_location='cpu')
    arcface_backbone.load_state_dict(state_dict)
    arcface_backbone.to(device)

    # 클래스 이름 로드
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # 분류기 정의
    embedding_size = 512
    class ArcFaceClassifier(nn.Module):
        def __init__(self, backbone, embedding_size, num_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            emb = self.backbone(x)  # [B, 512]
            out = self.classifier(emb)  # [B, num_classes]
            return out

    model = ArcFaceClassifier(arcface_backbone, embedding_size, num_classes).to(device)

    # 파인튜닝된 모델 가중치 로드
    model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    model.eval()

    return model, transform, class_names, device

def predict_image(model, img, transform, class_names, device):
    """
    이미지를 입력받아 모델을 통해 예측된 클래스를 반환하는 함수.

    Args:
        model (nn.Module): 로드된 모델.
        img (PIL.Image.Image): 입력 이미지.
        transform (transforms.Compose): 이미지 변환 파이프라인.
        class_names (list): 클래스 이름 리스트.
        device (torch.device): 사용 중인 디바이스.

    Returns:
        str: 예측된 클래스 이름.
    """
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 112, 112]
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0].item()]
