import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from huggingface_hub import hf_hub_download

# ✅ Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="SyedaBushra/C2D",  # Replace with your Hugging Face repo
    filename="ckpt_clothing_resnet50.pth"
)

# ✅ Load the checkpoint with weights_only=False
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# ✅ Extract only the model weights
model_state_dict = checkpoint['model']

# ✅ Initialize ResNet-50 model architecture (modify num_classes if necessary)
model = models.resnet50(num_classes=14)

# ✅ Rename keys if necessary (removing "encoder.module." prefix)
new_state_dict = {k.replace('encoder.module.', ''): v for k, v in model_state_dict.items()}

# ✅ Load model weights with `strict=False`
model.load_state_dict(new_state_dict, strict=False)
model.eval()  # Set model to evaluation mode

# ✅ Define class labels (Modify as needed)
class_labels = [
    "T-shirt", "Dress", "Jeans", "Shoes", "Hat", "Jacket", "Shirt",
    "Shorts", "Skirt", "Sweater", "Bag", "Socks", "Suit", "Gloves"
]

# ✅ Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ✅ Define inference function
def classify_image(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)  # Get model prediction
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = class_labels[predicted_class]
    
    return predicted_label  # Return the predicted clothing item name

# ✅ Create Gradio Interface
interface = gr.Interface(
    fn=classify_image,  # Function to call for inference
    inputs=gr.Image(type="pil"),  # User uploads an image
    outputs=gr.Text(),  # Output is a text label
    title="Clothing Classification Model",
    description="Upload an image of clothing to classify it into one of 14 categories."
)

# ✅ Run Gradio app locally
if __name__ == "__main__":
    interface.launch()
