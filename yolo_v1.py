# --- 1. Импорты и окружение ---
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. Загрузка Pascal VOC ---
transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
])

train_dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# --- 3. Подготовка данных для YOLOv1 ---
def prepare_yolo_targets(targets, S=7, B=2, C=20):
    batch_size = len(targets)
    target_tensor = torch.zeros(batch_size, S, S, B*5 + C)

    for b, target in enumerate(targets):
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            bbox = obj['bndbox']
            class_label = obj['name']

            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            if xmax <= xmin or ymax <= ymin:
                continue

            x_center = (xmin + xmax) / 2 / 448
            y_center = (ymin + ymax) / 2 / 448
            width = max((xmax - xmin) / 448, 1e-6)
            height = max((ymax - ymin) / 448, 1e-6)

            cell_x = min(int(x_center * S), S - 1)
            cell_y = min(int(y_center * S), S - 1)

            x_cell = x_center * S - cell_x
            y_cell = y_center * S - cell_y

            if target_tensor[b, cell_y, cell_x, 4] == 0:
                target_tensor[b, cell_y, cell_x, 0:5] = torch.tensor([x_cell, y_cell, width, height, 1])

                class_idx = VOC_CLASSES.index(class_label)
                target_tensor[b, cell_y, cell_x, B*5 + class_idx] = 1

    return target_tensor

# --- 4. Архитектура YOLOv1 ---
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S, self.B, self.C = S, B, C

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        # sigmoid для стабильности bbox и confidence
        x[..., :self.B*5] = torch.sigmoid(x[..., :self.B*5])
        return x

# --- 5. Функция потерь YOLOv1 ---
def yolo_loss(predictions, targets, S=7, B=2, C=20,
              lambda_coord=5, lambda_noobj=0.5):

    mse = nn.MSELoss(reduction='sum')
    obj_mask = targets[..., 4] > 0
    noobj_mask = targets[..., 4] == 0

    coord_loss = lambda_coord * mse(
        predictions[obj_mask][..., :2],
        targets[obj_mask][..., :2]
    )

    size_loss = lambda_coord * mse(
        torch.sqrt(torch.abs(predictions[obj_mask][..., 2:4]) + 1e-6),
        torch.sqrt(targets[obj_mask][..., 2:4] + 1e-6)
    )

    conf_loss_obj = mse(
        predictions[obj_mask][..., 4],
        targets[obj_mask][..., 4]
    )

    conf_loss_noobj = lambda_noobj * mse(
        predictions[noobj_mask][..., 4],
        targets[noobj_mask][..., 4]
    )

    class_loss = mse(
        predictions[obj_mask][..., 5:],
        targets[obj_mask][..., 5:]
    )

    total_loss = (coord_loss + size_loss +
                  conf_loss_obj + conf_loss_noobj +
                  class_loss)

    return total_loss / predictions.shape[0]

# --- 6. Обучение YOLOv1 ---
model = YOLOv1().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()

    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        targets_tensor = prepare_yolo_targets(targets).to(device)

        predictions = model(images)
        loss = yolo_loss(predictions, targets_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('🎉 Обучение YOLOv1 успешно завершено!')

from torchvision.ops import nms

def visualize_yolo_predictions_full(model, dataset, idx=0, S=7, B=2, C=20, threshold=0.2, nms_threshold=0.5):
    model.eval()
    img, target = dataset[idx]
    with torch.no_grad():
        predictions = model(img.unsqueeze(0).to(device))

    predictions = predictions.squeeze(0).cpu()
    boxes, scores, labels = [], [], []

    cell_size = 448 / S
    for i in range(S):
        for j in range(S):
            cell_pred = predictions[i, j]
            class_probs = cell_pred[B*5:]
            best_class_prob, class_label = torch.max(class_probs, dim=0)

            for b in range(B):
                conf = cell_pred[4 + b*5]
                final_conf = conf * best_class_prob  # общая уверенность (confidence × class_prob)

                if final_conf > threshold:
                    x_cell, y_cell, w, h = cell_pred[b*5:b*5+4]

                    x_center = (j + x_cell) * cell_size
                    y_center = (i + y_cell) * cell_size
                    width = w * 448
                    height = h * 448

                    xmin = max(x_center - width / 2, 0)
                    ymin = max(y_center - height / 2, 0)
                    xmax = min(x_center + width / 2, 448)
                    ymax = min(y_center + height / 2, 448)

                    boxes.append([xmin, ymin, xmax, ymax])
                    scores.append(final_conf.item())
                    labels.append(VOC_CLASSES[class_label])

    if len(boxes) == 0:
        print("Нет уверенных предсказаний. Уменьши порог.")
        return

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)

    keep = nms(boxes_tensor, scores_tensor, nms_threshold)

    plt.figure(figsize=(8, 8))
    plt.imshow(img.permute(1, 2, 0))
    ax = plt.gca()

    for idx in keep:
        xmin, ymin, xmax, ymax = boxes_tensor[idx]
        label = labels[idx]
        conf = scores_tensor[idx]

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='cyan', linewidth=2)
        ax.add_patch(rect)

        # Выводим класс объекта и уверенность
        ax.text(xmin, ymin - 10, f'{label}: {conf:.2f}',
                color='yellow', fontsize=12, weight='bold')

    plt.title('YOLOv1: bbox + классы после NMS')
    plt.axis('off')
    plt.show()

# Тестируем на изображении №0
visualize_yolo_predictions_full(
    model, train_dataset, idx=0, threshold=0.05, nms_threshold=0.5
)

