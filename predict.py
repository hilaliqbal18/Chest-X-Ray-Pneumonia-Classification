import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict(image_path, model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate the exact architecture used for training
    model = models.resnet18(weights=None) # We don't need ImageNet weights, we have our own!
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2) # 2 classes: NORMAL, PNEUMONIA
    )
    
    # Load your saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Preprocess the image (must match the validation transforms used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        
    classes = ['NORMAL', 'PNEUMONIA']
    prediction = classes[preds.item()]
    confidence = conf.item() * 100
    
    # Print the result nicely
    print("\n" + "="*40)
    print(f"🩺 Prediction: {prediction}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Pneumonia from Chest X-Ray")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to model weights")
    args = parser.parse_args()
    
    predict(args.image, args.model)