import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_DIR = './data/test_images/'
MODEL_WEIGHTS = "best_mit_b5_scse_model.pth"

# Defined Colors (RGB)
COLOR_MAP = {
    0: (0, 0, 0),       # Black (Background)
    1: (255, 0, 0),     # Red (Aggregation)
    2: (0, 255, 0),     # Green (Normal)
    3: (255, 255, 0),   # Yellow (Blur)
    4: (128, 0, 128),   # Purple (Abnormal)
    5: (0, 255, 255)    # Cyan (Hemo)
}

CLASSES = {
    0: "Background",
    1: "Aggregation",
    2: "Normal",
    3: "Blur",
    4: "Abnormal",
    5: "Hemo"
}

# ================= HELPER FUNCTIONS =================
def mask_to_rgb(mask, color_map):
    """Converts a class index mask (H, W) to an RGB image (H, W, 3)"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        # Mask where value equals class_id
        idx = (mask == class_id)
        rgb[idx] = color
    return rgb

# ================= CORE REPAIR LOGIC =================
def analyze_mask_with_repair(prediction_mask):
    """
    Input: Raw prediction mask (fragmented).
    Output: 
      1. stats: Dictionary of counts per class.
      2. final_mask: The repaired mask for visualization.
    """
    stats = {}
    
    # Canvas for the final repaired result
    final_mask = np.zeros_like(prediction_mask)
    
    # --- STRATEGY UPDATE ---
    # 1. Base Kernel: Fixes small pixel noise inside a blob
    kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # 2. Vertical Kernel: Specifically bridges vertical gaps
    # Increased height to (3, 25) to fix larger breaks. 
    # Width stays 3 to prevent merging neighbor capillaries.
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    
    min_area_threshold = 40  # Filter out small noise dots

    print("\n--- Analysis Report ---")
    
    for class_id, class_name in CLASSES.items():
        if class_id == 0: continue # Skip Background
        
        # 1. Get Binary Mask for this class
        binary_mask = np.uint8(prediction_mask == class_id) * 255
        
        # 2. REPAIR STEP A: Basic consolidation (fill small holes)
        step1 = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_base)
        
        # 3. REPAIR STEP B: Aggressive Vertical Closing (bridge gaps)
        repaired_binary = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, kernel_vertical)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(repaired_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        
        # 5. Filter Noise and Draw
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area_threshold:
                count += 1
                # Draw the valid, repaired object onto the final mask
                cv2.drawContours(final_mask, [cnt], -1, class_id, -1)
        
        stats[class_name] = count
        print(f"Class {class_name}: Detected {count}")

    return stats, final_mask

# ================= MODEL & PIPELINE =================
def load_model():
    print(f"🚀 Loading Model: MiT-B5 (scSE)...")
    model = smp.Unet(
        encoder_name="mit_b5",
        classes=6,
        decoder_attention_type="scse"
    ).to(DEVICE)
    
    if os.path.exists(MODEL_WEIGHTS):
        state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

def run_pipeline():
    # 1. Setup
    model = load_model()
    img_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    if not img_files: raise FileNotFoundError("No images in test folder")
    
    random_file = random.choice(img_files)
    # If you want to test a specific broken image, uncomment below:
    # random_file = "8_54890_5.jpg" 
    
    img_path = os.path.join(TEST_IMAGE_DIR, random_file)
    print(f"🔍 Analyzing Image: {random_file}")
    
    # 2. Preprocessing
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_raw = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. Repair & Count
    stats, repaired_mask = analyze_mask_with_repair(pred_mask)
    
    # 5. Visualization
    visualize_repair(image_rgb, pred_mask, repaired_mask, stats)

def visualize_repair(original, raw_mask, repaired_mask, stats):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Raw Prediction (Convert to RGB for correct colors)
    raw_rgb = mask_to_rgb(raw_mask, COLOR_MAP)
    axes[1].imshow(raw_rgb)
    axes[1].set_title("Raw Prediction (Fragmented)", fontsize=14)
    axes[1].axis('off')
    
    # Repaired Prediction (Convert to RGB for correct colors)
    repaired_rgb = mask_to_rgb(repaired_mask, COLOR_MAP)
    axes[2].imshow(repaired_rgb)
    
    # Create Title String from Stats
    stats_text = "Repaired & Filtered Results:\n"
    for k, v in stats.items():
        stats_text += f"{k}: {v}\n"
        
    axes[2].set_title(stats_text, fontsize=12, loc='left', family='monospace')
    axes[2].axis('off')
    
    # Create Legend
    patches = []
    for cls_id, color in COLOR_MAP.items():
        if cls_id == 0: continue
        c_norm = (color[0]/255, color[1]/255, color[2]/255)
        patches.append(mpatches.Patch(color=c_norm, label=CLASSES[cls_id]))
    
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize='large')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    print("✅ Visualization Generated.")

if __name__ == "__main__":
    run_pipeline()