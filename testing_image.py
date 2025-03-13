import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 

from solver_header import CaptchaDataset, transform, index_to_char

# Load dataset
train_dataset = CaptchaDataset(root_dir="final_training", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Get one batch of data
dataiter = iter(train_loader)
images, labels = next(dataiter)  # Get first batch

# Convert tensor image back to a displayable format
def show_transformed_image(image_tensor):
    image = image_tensor.squeeze(0)  # Remove batch dimension
    plt.imshow(image.numpy(), cmap="gray")  # Convert to numpy & plot
    plt.axis("off")  # Hide axes
    plt.show()

# Show first image in the batch
show_transformed_image(images[0])
print("Label:", labels[0].tolist())  # Show corresponding label

# Get actual text
predicted_chars = "".join(index_to_char[idx] for idx in labels[0].tolist())
print("CAPTCHA Text:", predicted_chars)