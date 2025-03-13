from solver_header import CaptchaCNN, CaptchaDataset, possible_chars, transform

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# Load dataset
train_dataset = CaptchaDataset(root_dir="final_training", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
num_classes = len(possible_chars)
captcha_length = 6
model = CaptchaCNN(num_classes=num_classes, captcha_length=captcha_length)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Works for classification
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the checkpoint
if os.path.isfile('captcha_solver.pth'):
    checkpoint = torch.load('captcha_solver.pth')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()  # Set model to training mode

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available

        optimizer.zero_grad()
        outputs = model(images)  # Shape: [batch_size, captcha_length, num_classes]
        
        # Compute loss for each character
        # Reshape outputs & labels: Flatten everything for batch processing
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad.mean()}")
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy (correctly predicted full CAPTCHAs)
        predicted = torch.argmax(outputs, dim=2)  # [batch_size, captcha_length]
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total * 100:.2f}%")

# Save trained model
# Save both model and optimizer
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': num_classes,
    'captcha_length': captcha_length
}, 'captcha_solver.pth')

print("Model training complete and saved.")