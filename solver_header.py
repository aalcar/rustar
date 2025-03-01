import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image

# captcha Convulutional Neural Network? idk what that means tbh.
# i asked chatgpt to make a model for me; i dont think it knew what it was doing
class CaptchaCNN(nn.Module):
    def __init__(self, num_classes = 62):  # 26 lower-case letters + 26 upper-case letters + 10 digits
        super(CaptchaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def predict_captcha(image_path):
    image = Image.open(image_path)  # Load the image
    image = transform(image).unsqueeze(0)  # Apply transformations & add batch dimension
    train_dataset = datasets.ImageFolder(root='training', transform=transform)

    with torch.no_grad():  # No need to calculate gradients
        model = CaptchaCNN(num_classes=len(train_dataset.classes))
        output = model(image)  # Get model prediction
        predicted_class = torch.argmax(output, dim=1).item()  # Get class index

    # Convert class index to character (assuming dataset classes are sorted)
    class_to_label = train_dataset.classes  # The class names from training
    predicted_label = class_to_label[predicted_class]
    
    return predicted_label

possible_chars = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J',
                  'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T',
                  'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9']

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])