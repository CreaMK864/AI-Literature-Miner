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

# FILTERING THRESHOLDS (The key to fixing your problem)
# Before: 30 -> Now: 100. Any object smaller than 100 pixels is ignored.
MIN_AREA_THRESHOLD = 300 

# Colors (RGB)
COLOR_MAP = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Aggregation (Red)
    2: (0, 255, 0),     # Normal (Green)
    3: (255, 255, 0),   # Blur (Yellow)
    4: (128, 0, 128),   # Abnormal (Purple)
    5: (0, 255, 255)    # Hemo (Cyan)
}

CLASSES = {
    0: "Background",
    1: "Aggregation",
    2: "Normal",
    3: "Blur",
    4: "Abnormal",
    5: "Hemo"
}

# ================= CORE LOGIC =================

def smart_post_processing(raw_mask):
    """
    Advanced pipeline with Strict Noise Filtering.
    """
    h, w = raw_mask.shape
    final_mask = np.zeros_like(raw_mask)
    
    # 1. Create Global Binary Mask (Foreground vs Background)
    binary_mask = (raw_mask > 0).astype(np.uint8) * 255
    
    # 2. NOISE REMOVAL STEP (Morphological Opening)
    # This erases tiny separate dots before we even start counting.
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # 3. Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    fragments = []
    
    # 4. Filter by Area (The Gatekeeper)
    for i in range(1, num_labels): # Skip background 0
        area = stats[i, cv2.CC_STAT_AREA]
        
        # KEY FIX: Strictly ignore anything smaller than MIN_AREA_THRESHOLD
        if area < MIN_AREA_THRESHOLD:
            continue
            
        x, y, w_rect, h_rect = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        center_x = centroids[i][0]
        
        fragments.append({
            'id': i,
            'center_x': center_x,
            'y_top': y,
            'y_bottom': y + h_rect,
            'area': area,
            'merged': False,
            'group_id': i
        })

    # 5. Geometric Stitching (Fixing broken capillaries)
    MAX_X_DIST = 15   # Increased slightly to catch more vertical breaks
    MAX_Y_GAP = 60    # Increased slightly
    
    fragments.sort(key=lambda k: k['y_top'])
    merge_map = {f['id']: f['id'] for f in fragments}
    
    for i in range(len(fragments)):
        f1 = fragments[i]
        for j in range(i + 1, len(fragments)):
            f2 = fragments[j]
            
            if f2['y_top'] - f1['y_bottom'] > MAX_Y_GAP:
                break
            
            # If they are vertically aligned (similar X)
            if abs(f1['center_x'] - f2['center_x']) < MAX_X_DIST:
                root_group = merge_map[f1['id']]
                merge_map[f2['id']] = root_group
                f1['y_bottom'] = max(f1['y_bottom'], f2['y_bottom'])

    # 6. Majority Voting & Drawing
    groups = {}
    for f in fragments:
        gid = merge_map[f['id']]
        if gid not in groups: groups[gid] = []
        groups[gid].append(f['id'])
        
    final_stats = {}
    
    for gid, member_ids in groups.items():
        # Get all pixels belonging to this merged group
        group_mask = np.isin(labels, member_ids)
        
        # Find dominant color
        pixel_values = raw_mask[group_mask]
        pixel_values = pixel_values[pixel_values > 0]
        
        if len(pixel_values) == 0: continue
            
        counts = np.bincount(pixel_values)
        dominant_class = np.argmax(counts)
        
        # Paint the group with the dominant color
        final_mask[group_mask] = dominant_class
        
        # Connect fragments visually with a line
        if len(member_ids) > 1:
            member_frags = [f for f in fragments if f['id'] in member_ids]
            member_frags.sort(key=lambda k: k['y_top'])
            for k in range(len(member_frags) - 1):
                pt1 = (int(member_frags[k]['center_x']), int(member_frags[k]['y_bottom']))
                pt2 = (int(member_frags[k+1]['center_x']), int(member_frags[k+1]['y_top']))
                # Draw thick line to bridge gap
                cv2.line(final_mask, pt1, pt2, int(dominant_class), thickness=6)

        # Update stats
        class_name = CLASSES.get(dominant_class, "Unknown")
        final_stats[class_name] = final_stats.get(class_name, 0) + 1

    return final_stats, final_mask

# ================= UTILS & RUNNER =================
def mask_to_rgb(mask, color_map):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb[mask == class_id] = color
    return rgb

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
    model = load_model()
    img_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    if not img_files: raise FileNotFoundError("No images found")
    
    random_file = random.choice(img_files)
    # random_file = "8_54890_5.jpg" # Fixed for testing specific broken images
    
    img_path = os.path.join(TEST_IMAGE_DIR, random_file)
    print(f"🔍 Analyzing Image: {random_file}")
    
    # Preprocess
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_raw = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # Process
    stats, refined_mask = smart_post_processing(pred_mask)
    
    # Visualize
    visualize_results(image_rgb, pred_mask, refined_mask, stats)

def visualize_results(original, raw_mask, refined_mask, stats):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 把圖變大一點方便看
    
    # 1. 原始圖
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    # 2. 原始預測
    raw_rgb = mask_to_rgb(raw_mask, COLOR_MAP)
    axes[1].imshow(raw_rgb)
    axes[1].set_title("Raw Prediction\n(Fragmented)", fontsize=16)
    axes[1].axis('off')
    
    # 3. 最終結果 (加上編號！)
    refined_rgb = mask_to_rgb(refined_mask, COLOR_MAP)
    
    # --- 新增功能：在圖上畫數字 ---
    # 我們再次找出輪廓，只是這次是為了畫圖
    vis_img = refined_rgb.copy()
    
    # 針對每一個類別畫輪廓和編號
    total_counts = {}
    
    for class_id, class_name in CLASSES.items():
        if class_id == 0: continue
        
        # 找出該類別的所有物體
        binary_mask = np.uint8(refined_mask == class_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        for cnt in contours:
            # 這裡不需要再過濾面積，因為 refined_mask 已經是過濾乾淨的結果
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 在血管中心寫上編號 (例如: N1, N2...)
                # N=Normal, A=Aggregation, H=Hemo...
                prefix = class_name[0] 
                label = f"{prefix}{count+1}"
                
                # 畫黑底白字，確保看得到
                cv2.putText(vis_img, label, (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                count += 1
        
        if count > 0:
            total_counts[class_name] = count

    axes[2].imshow(vis_img)
    
    # 更新標題顯示統計
    stats_text = "Final Indexed Analysis:\n"
    for k, v in total_counts.items():
        stats_text += f"{k}: {v}\n"
        
    axes[2].set_title(stats_text, fontsize=14, loc='left', family='monospace', fontweight='bold')
    axes[2].axis('off')
    
    # Legend
    patches = []
    for cls_id, color in COLOR_MAP.items():
        if cls_id == 0: continue
        c_norm = (color[0]/255, color[1]/255, color[2]/255)
        patches.append(mpatches.Patch(color=c_norm, label=CLASSES[cls_id]))
    
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize='large')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    save_path = "final_indexed_result.png"
    plt.savefig(save_path)
    print(f"✅ Saved result with IDs to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()