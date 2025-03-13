import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# captcha Convulutional Neural Network? idk what that means tbh.
class CaptchaCNN(nn.Module):
    def __init__(self, num_classes = 62, captcha_length = 6):  # 26 lower-case letters + 26 upper-case letters + 10 digits
        super(CaptchaCNN, self).__init__()
        self.num_classes = num_classes
        self.captcha_length = captcha_length

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes * captcha_length)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x.view(x.size(0), -1, self.num_classes)

possible_chars = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J',
                  'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T',
                  'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9']

# Index encoding/decoding
char_to_index = {char: idx for idx, char in enumerate(possible_chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None, captcha_length=6):
        self.root_dir = root_dir
        self.transform = transform
        self.captcha_length = captcha_length
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("L")  # Grayscale

        # Extract label from filename
        label_str = os.path.splitext(img_name)[0]  # Remove .png extension
        label = [char_to_index[char] for char in label_str]  # Convert characters to indices

        # Ensure label has correct length
        if len(label) < self.captcha_length:
            label += [0] * (self.captcha_length - len(label))  # Pad if too short
        elif len(label) > self.captcha_length:
            label = label[:self.captcha_length]  # Trim if too long

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)  # Return image + label tensor

def predict_captcha(image_path):
    image = Image.open(image_path)  # Load the image
    image = transform(image).unsqueeze(0)  # Apply transformations & add batch dimension
    train_dataset = datasets.ImageFolder(root='final_training', transform=transform)

    with torch.no_grad():  # No need to calculate gradients
        model = CaptchaCNN(num_classes=len(train_dataset.classes))
        model.load_state_dict(torch.load('captcha_solver.pth')['model_state_dict'])
        output = model(image)  # Get model prediction
        predicted_class = torch.argmax(output, dim=1).item()  # Get class index

    # Convert class index to character (assuming dataset classes are sorted)
    class_to_label = train_dataset.classes  # The class names from training
    predicted_label = class_to_label[predicted_class]
    
    return predicted_label

def predict_captcha(image_path, model_path='captcha_solver.pth'):
    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = len(possible_chars)
    captcha_length = 6  # Ensure this matches training
    
    model = CaptchaCNN(num_classes=num_classes, captcha_length=captcha_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Grayscale
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    with torch.no_grad():  
        output = model(image)  # Model predicts [batch_size, captcha_length, num_classes]
        predicted_indices = torch.argmax(output, dim=2).squeeze(0)  # Get most likely class for each character
        predicted_text = ''.join([possible_chars[idx] for idx in predicted_indices])  # Convert indices to text

    return predicted_text

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda img: img.resize((64, 64), Image.LANCZOS)),  # Resize first
    transforms.ToTensor()
])