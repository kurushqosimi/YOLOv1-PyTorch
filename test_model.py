import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
from model import YOLOv1
from torchvision.ops import nms

# загружаем модель
device = torch.device('cpu')
model = YOLOv1().to(device)
model.load_state_dict(torch.load('yolov1_model.pth', map_location=device))
model.eval()

transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
])

dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
    'sheep', 'sofa', 'train', 'tvmonitor']

def visualize(idx, threshold=0.1, nms_threshold=0.5):
    img, _ = dataset[idx]
    with torch.no_grad():
        predictions = model(img.unsqueeze(0))

    predictions = predictions.squeeze(0).cpu()
    boxes, scores, labels = [], [], []

    cell_size = 448 / 7
    for i in range(7):
        for j in range(7):
            cell_pred = predictions[i, j]
            class_probs = cell_pred[10:]
            best_class_prob, class_label = torch.max(class_probs, dim=0)

            for b in range(2):
                conf = cell_pred[4 + b*5]
                final_conf = conf * best_class_prob
                if final_conf > threshold:
                    x_cell, y_cell, w, h = cell_pred[b*5:b*5+4]
                    x_center = (j + x_cell) * cell_size
                    y_center = (i + y_cell) * cell_size
                    width, height = w * 448, h * 448

                    xmin, ymin = max(x_center - width / 2, 0), max(y_center - height / 2, 0)
                    xmax, ymax = min(x_center + width / 2, 448), min(y_center + height / 2, 448)

                    boxes.append([xmin, ymin, xmax, ymax])
                    scores.append(final_conf.item())
                    labels.append(VOC_CLASSES[class_label])

    if not boxes:
        print('Нет объектов. Снизьте порог!')
        return

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep = nms(boxes_tensor, scores_tensor, nms_threshold)

    plt.figure(figsize=(8, 8))
    plt.imshow(img.permute(1,2,0))
    ax = plt.gca()

    for idx in keep:
        xmin, ymin, xmax, ymax = boxes_tensor[idx]
        label, conf = labels[idx], scores_tensor[idx]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='cyan', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin-10, f'{label}: {conf:.2f}', color='yellow', fontsize=12, weight='bold')

    plt.axis('off')
    plt.title('Тест YOLOv1 локально')
    plt.show()

visualize(0)