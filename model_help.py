import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['Banana',
 'Lemon',
 'Lulo',
 'Mango',
 'Orange',
 'Strawberry',
 'Tamarillo',
 'Tomato',
 'Banana',
 'Lemon',
 'Lulo',
 'Mango',
 'Orange',
 'Strawberry',
 'Tamarillo',
 'Tomato']

class FreshHarvestRestNetCNN(nn.Module):
    def __init__(self,num_classes=16,dropout_rate=.5):
        super().__init__()
        self.network = models.resnet50(weights='DEFAULT')
        for param in self.network.parameters():
            param.requires_grad = False

        for param in self.network.layer4.parameters():
            param.requires_grad = True

        self.network.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.network.fc.in_features,num_classes)
        )

    def forward(self,x):
        x = self.network(x)
        return x

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = FreshHarvestRestNetCNN()
        # 👇 This ensures it loads correctly on CPU-only systems
        trained_model.load_state_dict(
            torch.load("model/saved_model.pth", map_location=torch.device("cpu"))
        )
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
