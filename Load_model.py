import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
# import gradio as gr
# from huggingface_hub import hf_hub_download


# Path to the pretrained model
model_path = r"C:\Users\SyedaBushraFatima\Documents\AI in modern Software Development\Assignment 2 HuggingFace\C2D\pretrained\ckpt_clothing_resnet50.pth"

# Load the checkpoint with weights_only=False
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# Print the available keys in the checkpoint
# print("Checkpoint Keys:", checkpoint.keys())

# Extract only the model state_dict
model_state_dict = checkpoint['model']  # Extract only the model weights

# Load the ResNet-50 model architecture
model = models.resnet50(num_classes=14)  # Adjust `num_classes` if necessary

# Rename keys if necessary
new_state_dict = {k.replace('encoder.module.', ''): v for k, v in model_state_dict.items()}

# Load the adjusted state dictionary
model.load_state_dict(new_state_dict, strict=False)  # Use `strict=False` to ignore extra/missing keys

# Set model to evaluation mode
model.eval()

# Print model summary
# print(model)




# Create a random tensor input with batch size 1, 3 color channels, and 224x224 image size
dummy_input = torch.randn(1, 3, 224, 224)

# Perform inference
with torch.no_grad():
    output = model(dummy_input)

# Print the raw output from the model
# print("Model Output:", output)
# print("Predicted Class Index:", torch.argmax(output, dim=1).item())


# Load an actual image (change the path to your test image)
image_path = r"C:\Users\SyedaBushraFatima\Documents\AI in modern Software Development\Assignment 2 HuggingFace\C2D\sample_image.jpg"
image = Image.open(image_path).convert("RGB")  # Convert to RGB

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations and add batch dimension
input_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get predicted class
predicted_class = torch.argmax(output, dim=1).item()
print("Predicted Class Index:", predicted_class)


# Assuming you have 14 classes in your dataset, define labels (modify as needed)
class_labels = [
    "T-shirt", "Dress", "Jeans", "Shoes", "Hat", "Jacket", "Shirt",
    "Shorts", "Skirt", "Sweater", "Bag", "Socks", "Suit", "Gloves"
]

# Get the predicted class name
predicted_label = class_labels[predicted_class]
print("Predicted Clothing Item:", predicted_label)

import torch
from PIL import Image
import torchvision.transforms as transforms

# Define the image path
image_path = r"C:\Users\SyedaBushraFatima\Documents\AI in modern Software Development\Assignment 2 HuggingFace\C2D\sample_image.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")  # Convert to RGB if not already

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Apply transformations
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    output = model(input_tensor)  # Get predictions

# Get predicted class index
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted Class Index: {predicted_class}")


import matplotlib.pyplot as plt

# Display the image
plt.imshow(image)
plt.axis("off")  # Hide axis
plt.title(f"Predicted: {predicted_label}")  # Show predicted label
plt.show()