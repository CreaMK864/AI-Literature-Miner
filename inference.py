import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS = "best_mit_b5_scse_model.pth" 
MIN_AREA_THRESHOLD = 300  # Filter out noise smaller than 300 pixels

CLASSES = {
    0: "Background", 
    1: "Aggregation", 
    2: "Normal",
    3: "Blur", 
    4: "Abnormal", 
    5: "Hemo"
}

# Neon colors for better visibility on skin overlay
COLOR_MAP = {
    0: (0, 0, 0),       
    1: (255, 0, 0),     # Red (Aggregation)
    2: (0, 255, 0),     # Green (Normal)
    3: (255, 255, 0),   # Yellow (Blur)
    4: (255, 0, 255),   # Magenta (Abnormal - High Contrast)
    5: (0, 255, 255)    # Cyan (Hemo)
}

_model = None

def get_model():
    """Singleton model loader to prevent reloading on every run."""
    global _model
    if _model is not None: return _model
    
    print("🚀 Loading Model...")
    model = smp.Unet(
        encoder_name="mit_b5", 
        classes=6, 
        decoder_attention_type="scse"
    ).to(DEVICE)
    
    if os.path.exists(MODEL_WEIGHTS):
        state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        # Handle state_dict nesting if present
        if 'model_state_dict' in state_dict: 
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _model = model
        return model
    else:
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

def process_image(image_file):
    """
    Main Pipeline: Load -> Preprocess -> Infer -> Post-process -> Overlay
    """
    # 1. Load Image from Streamlit
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # 2. Preprocessing
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    model = get_model()
    with torch.no_grad():
        logits = model(input_tensor)
        pred_raw = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # 4. Post-processing (Repair broken loops + Count)
    stats, final_mask = smart_post_processing(pred_mask)
    
    # 5. Visualization (OVERLAY Logic)
    # Pass original RGB image to draw mask on top of it
    result_img = draw_result_on_image(image_rgb, final_mask)
    
    return image_rgb, result_img, stats

def smart_post_processing(raw_mask):
    """
    Advanced logic:
    1. Filter noise.
    2. Stitch vertical fragments.
    3. Majority voting for color consistency.
    """
    h, w = raw_mask.shape
    
    # Binary mask for object detection
    binary_mask = (raw_mask > 0).astype(np.uint8) * 255
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_clean)
    
    num_labels, labels, stats_cv, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    fragments = []
    
    # Collect valid fragments
    for i in range(1, num_labels):
        area = stats_cv[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA_THRESHOLD: continue
        
        x, y, w_rect, h_rect = stats_cv[i, cv2.CC_STAT_LEFT], stats_cv[i, cv2.CC_STAT_TOP], stats_cv[i, cv2.CC_STAT_WIDTH], stats_cv[i, cv2.CC_STAT_HEIGHT]
        fragments.append({
            'id': i, 'center_x': centroids[i][0], 
            'y_top': y, 'y_bottom': y + h_rect
        })

    # Vertical Stitching Logic
    MAX_X_DIST = 15
    MAX_Y_GAP = 80 
    fragments.sort(key=lambda k: k['y_top'])
    merge_map = {f['id']: f['id'] for f in fragments}
    
    for i in range(len(fragments)):
        f1 = fragments[i]
        for j in range(i + 1, len(fragments)):
            f2 = fragments[j]
            # Stop looking if vertical gap is too large
            if f2['y_top'] - f1['y_bottom'] > MAX_Y_GAP: break
            
            # Merge if aligned vertically
            if abs(f1['center_x'] - f2['center_x']) < MAX_X_DIST:
                merge_map[f2['id']] = merge_map[f1['id']]
                f1['y_bottom'] = max(f1['y_bottom'], f2['y_bottom'])

    # Reconstruct Final Mask
    final_stats = {}
    final_mask = np.zeros_like(raw_mask)
    groups = {}
    
    for f in fragments:
        gid = merge_map[f['id']]
        if gid not in groups: groups[gid] = []
        groups[gid].append(f['id'])
        
    for gid, member_ids in groups.items():
        group_mask = np.isin(labels, member_ids)
        pixel_values = raw_mask[group_mask]
        pixel_values = pixel_values[pixel_values > 0]
        
        if len(pixel_values) == 0: continue
        
        # Determine dominant color
        counts = np.bincount(pixel_values)
        dominant_class = np.argmax(counts)
        
        # Paint mask
        final_mask[group_mask] = dominant_class
        
        # Visual Stitching (Draw lines between merged fragments)
        if len(member_ids) > 1:
            member_frags = [f for f in fragments if f['id'] in member_ids]
            member_frags.sort(key=lambda k: k['y_top'])
            for k in range(len(member_frags) - 1):
                pt1 = (int(member_frags[k]['center_x']), int(member_frags[k]['y_bottom']))
                pt2 = (int(member_frags[k+1]['center_x']), int(member_frags[k+1]['y_top']))
                cv2.line(final_mask, pt1, pt2, int(dominant_class), thickness=6)
        
        # Update Stats
        class_name = CLASSES.get(dominant_class, "Unknown")
        final_stats[class_name] = final_stats.get(class_name, 0) + 1
            
    return final_stats, final_mask

def draw_result_on_image(original_image, mask):
    """
    Input: Original RGB Image, Segmentation Mask
    Output: Image with colored mask OVERLAY (Alpha Blending)
    """
    # Create a solid color layer
    color_mask = np.zeros_like(original_image)
    
    # Fill colors
    for cls_id, color in COLOR_MAP.items():
        if cls_id == 0: continue
        color_mask[mask == cls_id] = color
        
    # --- Overlay Logic ---
    # Identify where the mask is present
    mask_bool = (mask > 0)
    final_img = original_image.copy()
    
    # Only blend pixels where the mask exists (preserves background brightness)
    # 0.6 * Original + 0.5 * Color
    final_img[mask_bool] = cv2.addWeighted(original_image, 0.6, color_mask, 0.5, 0)[mask_bool]

    # Add Text Labels (ID tags)
    for cls_id, class_name in CLASSES.items():
        if cls_id == 0: continue
        
        binary_mask = np.uint8(mask == cls_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                label = f"{class_name[0]}{count+1}"
                
                # Black Outline (Shadow) for readability
                cv2.putText(final_img, label, (cX-10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                # White Text
                cv2.putText(final_img, label, (cX-10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                count += 1
    return final_img