from huggingface_hub import hf_hub_download
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="your-huggingface-username/your-model-name",
    filename="ckpt_clothing_resnet50.pth"
)

# Load the model
model = models.resnet50(num_classes=14)  # Adjust num_classes if needed
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Define inference function
def classify_image(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    return output.argmax(dim=1).item()
