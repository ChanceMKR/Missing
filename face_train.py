import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def main():
    # (1) arcface_torch 백본 import
    # InsightFace의 arcface_torch/backbones/ 폴더를 path에 추가 (필요 시 경로 수정)
    sys.path.append('insightface/recognition/arcface_torch/backbones')
    from iresnet import iresnet50  # iresnet100 등도 가능

    # 2. 하이퍼파라미터 & 디바이스 설정
    data_dir = 'data/faces'     # 얼굴 이미지를 폴더 구조로 저장한 경로
    backbone_path = 'weights/backbone.pth'  # 사전 학습된 ArcFace 백본 가중치
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-3

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    # 3. 전처리 (ArcFace 표준: 112×112 사용)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        # ArcFace/InsightFace 백본은 보통 BGR -> [ -1~1 ] 정규화 등을 사용하지만,
        # PyTorch에서는 RGB 순으로 읽으므로, 아래와 같은 정규화를 적용하는 경우가 많습니다.
        # (중요) pretrained 모델마다 요구되는 Normalize가 다를 수 있으니, 모델에 맞춰 조정하세요.
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 4. 데이터셋 (ImageFolder) & DataLoader
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print("인물(클래스) 목록:", class_names)

    # (선택) Train/Val 분할
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)


    # 5. ArcFace 백본(iResNet50) 불러오기
    # iresnet50(pretrained=False) 로드 후, 사전 학습된 backbone.pth를 가져옴
    arcface_backbone = iresnet50(pretrained=False, fp16=False)  # fp16=True 시 half precision
    state_dict = torch.load(backbone_path, map_location='cpu')
    arcface_backbone.load_state_dict(state_dict)
    arcface_backbone.to(device)

    # (옵션) 백본 완전 동결(freeze)하고 싶다면
    # for param in arcface_backbone.parameters():
    #     param.requires_grad = False

    # iResNet50은 입력(112×112) -> 출력(512차원 임베딩)이 기본
    embedding_size = 512

    # 6. 분류기(Linear) 얹기
    class ArcFaceClassifier(nn.Module):
        def __init__(self, backbone, embedding_size, num_classes):
            super().__init__()
            self.backbone = backbone
            # 여기서는 ArcFace의 마지막 'FC'가 아니라 임베딩만 추출하도록 사용하므로,
            # 분류 용도로 새 Linear Layer를 붙임
            self.classifier = nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            # x: [B, 3, 112, 112]
            emb = self.backbone.forward(x)   # [B, 512] 임베딩
            out = self.classifier(emb)       # [B, num_classes]
            return out

    model = ArcFaceClassifier(arcface_backbone, embedding_size, num_classes).to(device)

    # 7. 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 8. 학습 함수 정의
    def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")

            # ---------- [Train] ----------
            model.train()
            train_loss = 0.0
            train_correct = 0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)         # 분류 결과 [B, num_classes]
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * imgs.size(0)
                train_correct += torch.sum(preds == labels)

            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = train_correct.double() / len(train_loader.dataset)

            # ---------- [Validation] ----------
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * imgs.size(0)
                    val_correct += torch.sum(preds == labels)

            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_correct.double() / len(val_loader.dataset)

            print(f"  [Train] Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f}")
            print(f"  [Val]   Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.4f}")

            # 최고 정확도 업데이트 시 모델 저장
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), "best_arcface_finetune.pt")
                print("  >> Best model saved!")

        print(f"\n최고 검증 정확도: {best_acc:.4f}")

    # 9. 학습 실행
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)

    # 추론 예시
    model.load_state_dict(torch.load("best_arcface_finetune.pt"))
    model.eval()

main()