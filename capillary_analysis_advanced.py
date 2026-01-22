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

# ================= 配置區域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_DIR = './data/test_images/'
MODEL_WEIGHTS = "best_mit_b5_scse_model.pth"

# 顏色定義 (RGB)
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

# ================= 核心：先進後處理算法 =================

def smart_post_processing(raw_mask):
    """
    先進後處理流程：
    1. 噪點過濾
    2. 幾何拼接 (修復斷層，不合併鄰居)
    3. 多數決投票 (統一顏色)
    """
    h, w = raw_mask.shape
    
    # 1. 初始化一個全黑的畫布
    final_mask = np.zeros_like(raw_mask)
    
    # 2. 建立二值化 Mask (只要不是背景都算前景)
    # 用來找出所有的血管碎片，不管它是什麼顏色
    binary_mask = (raw_mask > 0).astype(np.uint8) * 255
    
    # 3. 基礎去噪 (去除極小雜點)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # 4. 找出所有獨立的連通區域 (Component Analysis)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 儲存有效的血管碎片信息
    # 格式: {'id': label_id, 'x': center_x, 'y_bottom': y+h, 'y_top': y, 'bbox': ...}
    fragments = []
    
    min_area = 30 # 最小面積閾值
    
    for i in range(1, num_labels): # 跳過背景 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
            
        x, y, w_rect, h_rect = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        center_x = centroids[i][0]
        
        fragments.append({
            'id': i,
            'center_x': center_x,
            'y_top': y,
            'y_bottom': y + h_rect,
            'area': area,
            'merged': False, # 標記是否已被合併
            'group_id': i    # 初始時自己一組
        })

    # 5. 【關鍵技術】幾何拼接 (Geometric Stitching)
    # 邏輯：如果在 X 軸上很接近，且在 Y 軸上斷裂處很近，則視為同一條
    
    # 參數設定
    MAX_X_DIST = 10   # X 軸允許的最大偏差 (像素)
    MAX_Y_GAP = 50    # Y 軸允許的最大斷裂距離 (像素)
    
    # 依據 Y 軸位置排序，從上到下處理
    fragments.sort(key=lambda k: k['y_top'])
    
    # 建立合併映射表 (Union-Find 概念的簡化版)
    merge_map = {f['id']: f['id'] for f in fragments}
    
    for i in range(len(fragments)):
        f1 = fragments[i]
        
        # 往後找可能的拼接對象
        for j in range(i + 1, len(fragments)):
            f2 = fragments[j]
            
            # 如果 f2 的頂部已經離 f1 底部太遠，後面的更不用看了 (因為已經按 Y 排序)
            if f2['y_top'] - f1['y_bottom'] > MAX_Y_GAP:
                break
                
            # 檢查 X 軸對齊程度 (是否在同一垂直線上)
            if abs(f1['center_x'] - f2['center_x']) < MAX_X_DIST:
                # 找到匹配！合併它們
                # 將 f2 的組別設為 f1 的組別
                root_group = merge_map[f1['id']]
                merge_map[f2['id']] = root_group
                
                # 更新 f1 的底部位置，以便能繼續往下接更下面的碎片
                f1['y_bottom'] = max(f1['y_bottom'], f2['y_bottom'])

    # 6. 【關鍵技術】多數決投票 (Majority Voting)
    # 根據合併後的組別，重新繪製 Mask
    
    # 將 fragment 依據 group_id 分組
    groups = {}
    for f in fragments:
        gid = merge_map[f['id']]
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(f['id'])
        
    final_stats = {}
    
    for gid, member_ids in groups.items():
        # 創建這個組別的 Mask
        group_mask = np.isin(labels, member_ids)
        
        # 在原始預測圖中，找出這個區域涵蓋的所有像素類別
        # raw_mask[group_mask] 會取出該區域所有像素的類別值
        pixel_values = raw_mask[group_mask]
        
        # 過濾掉 0 (背景)，雖然理論上 binary mask 已經過濾了，保險起見
        pixel_values = pixel_values[pixel_values > 0]
        
        if len(pixel_values) == 0:
            continue
            
        # 統計出現最多次的類別 (Mode)
        counts = np.bincount(pixel_values)
        dominant_class = np.argmax(counts)
        
        # 繪製到最終 Mask 上：
        # 這裡有兩種畫法：
        # A. 只畫原本的碎片 (保持斷裂但顏色統一)
        # B. 用線把碎片連起來 (修復斷裂) -> 我們選 B
        
        # 找出該組別所有碎片的輪廓並畫上去
        # 為了連接斷裂處，我們計算這些碎片的 Convex Hull (凸包) 或者直接畫線
        # 簡單做法：分別畫出每個碎片，然後如果是同一組，畫一條線連接它們的重心
        
        # 先畫原始碎片，統一顏色
        final_mask[group_mask] = dominant_class
        
        # 再畫連接線 (修復視覺斷層)
        if len(member_ids) > 1:
            member_frags = [f for f in fragments if f['id'] in member_ids]
            member_frags.sort(key=lambda k: k['y_top'])
            for k in range(len(member_frags) - 1):
                pt1 = (int(member_frags[k]['center_x']), int(member_frags[k]['y_bottom']))
                pt2 = (int(member_frags[k+1]['center_x']), int(member_frags[k+1]['y_top']))
                # 畫一條粗線連接
                cv2.line(final_mask, pt1, pt2, int(dominant_class), thickness=5)

        # 統計最終數量
        class_name = CLASSES.get(dominant_class, "Unknown")
        final_stats[class_name] = final_stats.get(class_name, 0) + 1

    return final_stats, final_mask

# ================= 模型載入與輔助函式 =================
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
    # 1. Setup
    model = load_model()
    img_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    if not img_files: raise FileNotFoundError("No images in test folder")
    
    random_file = random.choice(img_files)
    # random_file = "8_54890_5.jpg" # 如果要固定測試某張圖
    
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

    # 4. Advanced Post-Processing
    stats, refined_mask = smart_post_processing(pred_mask)
    
    # 5. Visualization
    visualize_results(image_rgb, pred_mask, refined_mask, stats)

def visualize_results(original, raw_mask, refined_mask, stats):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Raw Prediction
    raw_rgb = mask_to_rgb(raw_mask, COLOR_MAP)
    axes[1].imshow(raw_rgb)
    axes[1].set_title("Raw Prediction\n(Mixed Colors & Fragments)", fontsize=14)
    axes[1].axis('off')
    
    # Refined Prediction
    refined_rgb = mask_to_rgb(refined_mask, COLOR_MAP)
    axes[2].imshow(refined_rgb)
    
    # Generate Stat Text
    stats_text = "Advanced Analysis Result:\n"
    for k, v in stats.items():
        stats_text += f"{k}: {v}\n"
        
    axes[2].set_title(stats_text, fontsize=12, loc='left', family='monospace', fontweight='bold')
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
    save_path = "advanced_result.png"
    plt.savefig(save_path)
    print(f"✅ Visualization Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()