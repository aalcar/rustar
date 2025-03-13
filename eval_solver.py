from solver_header import predict_captcha
import random
import os

# Set path to your validation images
validation_dir = "final_training"
all_images = [f for f in os.listdir(validation_dir) if f.endswith('.png')]

# Pick a random image to test
image_name = random.choice(all_images)
image_path = os.path.join(validation_dir, image_name)
actual_text = os.path.splitext(image_name)[0]  # Extract text from filename

predicted_text = predict_captcha(image_path)

print(f"Predicted CAPTCHA: {predicted_text}")
print(f"Actual CAPTCHA: {actual_text}")
print(f"Correct: {predicted_text == actual_text}")