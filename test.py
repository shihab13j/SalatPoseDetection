import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Force CPU
device = torch.device('cpu')

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 5  # Background + chair + person + table
model = get_model(num_classes)

# Load model checkpoint on CPU
model.load_state_dict(torch.load("D:/Dataset/Salat Custom/Salatfasterrcnn_resnet50_epoch_3.pth", map_location=device))
model.to(device)
model.eval()

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)

image_path = "D:/Dataset/Salat Custom/train/salat mix1.jpg"
image_tensor = prepare_image(image_path)

with torch.no_grad():
    prediction = model(image_tensor)

COCO_CLASSES = {
    0: "Background", 
    1: "Stand", 
    2: "Sit", 
    3: "Ruku",
    4: "Sijdah"
}


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

def draw_boxes(image, prediction, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    threshold = 0.5
    plt.figure(figsize=fig_size)
    plt.imshow(image)
    
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                              linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.show()

draw_boxes(Image.open(image_path), prediction, fig_size=(12, 10))
