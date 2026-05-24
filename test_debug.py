import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Evaluating on:", device)

mtcnn = MTCNN(keep_all=False, device=device)

# --- Load ViT ---
model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = AutoImageProcessor.from_pretrained(model_id)
vit_model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
vit_model.eval()

# --- Load Local EfficientNet ---
eff_model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=2)
eff_model.load_state_dict(torch.load("model_best.pth", map_location=device))
eff_model.to(device)
eff_model.eval()

eff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def debug_ensemble(image_path):
    print(f"\n======================================")
    print(f"DEBUGGING INFERENCE OVER: {image_path}")
    print(f"======================================")
    
    image = Image.open(image_path).convert("RGB")
    
    # Defaults
    vit_image = image  # Default to raw image
    eff_image = image  # Default to raw image
    
    boxes, probs = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        # 1. Zero-Margin Crop (Exactly matches how extract_faces.py generated the EfficientNet training dataset)
        bleft, btop, bright, bbottom = box[0], box[1], box[2], box[3]
        eff_image = image.crop((int(bleft), int(btop), int(bright), int(bbottom)))
        
        # 2. 15% Padded Crop (for ViT, if needed)
        w, h = box[2] - box[0], box[3] - box[1]
        b1, b2 = max(0, box[0] - w * 0.15), max(0, box[1] - h * 0.15)
        b3, b4 = min(image.width, box[2] + w * 0.15), min(image.height, box[3] + h * 0.15)
        vit_image = image.crop((int(b1), int(b2), int(b3), int(b4)))
    else:
        print(">> WARNING: MTCNN found no face! Both models falling back to raw image.")

    print("\n--- 1. CLOUD ViT DIAGNOSTIC ---")
    inputs = processor(images=vit_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        model_probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
    labels = vit_model.config.id2label
    vit_fake_prob = 0.0
    vit_real_prob = 0.0
    for idx, label_name in labels.items():
        prob = model_probs[0][idx].item()
        l = label_name.lower()
        print(f">> IDX [{idx}] ({label_name}): {prob * 100:.2f}%")
        if 'fake' in l or 'deepfake' in l or 'spoof' in l:
            vit_fake_prob += prob
        elif 'real' in l or 'pristine' in l:
            vit_real_prob += prob
            
    print(f">> ViT Final Fake Score: {vit_fake_prob * 100:.2f}%")
    print(f">> ViT Final Real Score: {vit_real_prob * 100:.2f}%")

    print("\n--- 2. LOCAL EFFICIENTNET DIAGNOSTIC ---")
    eff_input = eff_transform(eff_image).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_outputs = eff_model(eff_input)
        eff_probs = torch.nn.functional.softmax(eff_outputs, dim=1)
        
    eff_fake_prob = eff_probs[0][0].item()  # 0 is Fake
    eff_real_prob = eff_probs[0][1].item()  # 1 is Real
    print(f">> Local Model FAKE (IDX 0): {eff_fake_prob * 100:.2f}%")
    print(f">> Local Model REAL (IDX 1): {eff_real_prob * 100:.2f}%")

    print("\n--- 3. MATHEMATICAL ENSEMBLE ---")
    # Current app.py weighting
    final_fake_prob = (0.7 * vit_fake_prob) + (0.3 * eff_fake_prob)
    final_real_prob = (0.7 * vit_real_prob) + (0.3 * eff_real_prob)
    
    print(f"> Current Output [70% ViT / 30% Eff]: FAKE={final_fake_prob*100:.2f}%, REAL={final_real_prob*100:.2f}%")
    
    # Proposed balanced weighting
    bal_fake = (0.5 * vit_fake_prob) + (0.5 * eff_fake_prob)
    bal_real = (0.5 * vit_real_prob) + (0.5 * eff_real_prob)
    print(f"> Balanced Output [50% ViT / 50% Eff]: FAKE={bal_fake*100:.2f}%, REAL={bal_real*100:.2f}%")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debug_ensemble(sys.argv[1])
    else:
        print("Please provide an image path: python test_debug.py <path_to_image>")
