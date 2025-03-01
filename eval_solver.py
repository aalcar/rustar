from solver_header import CaptchaCNN, possible_chars, predict_captcha

import torch

# Load the model
model = CaptchaCNN(num_classes=62)  # Ensure this matches the training setup
model.load_state_dict(torch.load('captcha_solver.pth'))
model.eval()  # Set the model to evaluation mode

total = 62
correct = 0
for char in possible_chars:
    image_path = f"training/{char}/image2.png"
    predicted_character = predict_captcha(image_path)

    print(f"Predicted character: {predicted_character}")
    print(f"Actual Character: {char} \n")

    if predicted_character == char:
        correct += 1

print(f"[{correct}/{total}]")