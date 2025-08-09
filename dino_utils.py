# grounding_utils.py
import os
import json
import cv2
import torch
from PIL import Image
from groundingdino.util.inference import load_model, predict, annotate
from torchvision.ops import box_iou
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import time

# === 依赖函数 ===
def xyxy_to_tensor(box):
    return torch.tensor([[box[0][0], box[0][1], box[1][0], box[1][1]]], dtype=torch.float)

def expand_box_with_min_area(x1, y1, x2, y2, scale, W, H, min_area=224 * 224):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    # 保证最小面积
    area = w * h
    if area < min_area:
        ratio = (min_area / area) ** 0.5
        w *= ratio
        h *= ratio

    # 保证宽高不小于224
    w = max(w, 224)
    h = max(h, 224)

    # 计算新框坐标
    nx1 = cx - w / 2
    nx2 = cx + w / 2
    ny1 = cy - h / 2
    ny2 = cy + h / 2

    # 限制框不要超过边界，优先平移框
    # 横向调整
    if nx1 < 0:
        shift = -nx1
        nx1 = 0
        nx2 += shift
        if nx2 > W - 1:
            nx2 = W - 1
            nx1 = max(nx2 - w, 0)
    if nx2 > W - 1:
        shift = nx2 - (W - 1)
        nx2 = W - 1
        nx1 -= shift
        if nx1 < 0:
            nx1 = 0
            nx2 = min(nx1 + w, W - 1)

    # 纵向调整
    if ny1 < 0:
        shift = -ny1
        ny1 = 0
        ny2 += shift
        if ny2 > H - 1:
            ny2 = H - 1
            ny1 = max(ny2 - h, 0)
    if ny2 > H - 1:
        shift = ny2 - (H - 1)
        ny2 = H - 1
        ny1 -= shift
        if ny1 < 0:
            ny1 = 0
            ny2 = min(ny1 + h, H - 1)

    # 最后转换成整数坐标
    nx1, ny1, nx2, ny2 = int(nx1), int(ny1), int(nx2), int(ny2)

    return nx1, ny1, nx2, ny2


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def adjust_box_to_original(box, offset, crop_size):
    w_crop, h_crop = crop_size
    xyxy_norm = cxcywh_to_xyxy(box)
    x1 = xyxy_norm[0] * w_crop
    y1 = xyxy_norm[1] * h_crop
    x2 = xyxy_norm[2] * w_crop
    y2 = xyxy_norm[3] * h_crop
    return [[int(x1 + offset[0]), int(y1 + offset[1])],
            [int(x2 + offset[0]), int(y2 + offset[1])]]

def fix_path(path_str, root_dir):
    normalized_path = path_str.replace("\\", "/")
    return os.path.abspath(os.path.join(root_dir, normalized_path))

def preprocess_pil_image(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(pil_img).unsqueeze(0).float()
    return pil_img, image_tensor



def run_groundingdino_on_image(
    rel_img_path: str,
    question: str,
    orig_box: list,  # [[x1, y1], [x2, y2]]
    model,
    device,
    root_dir,
    box_threshold=0.4,
    text_threshold=0.4,
    iou_threshold=0.6,
    enlarge_ratio=2.0,
    min_crop_size=224
):
    try:
        # === 修正路径 ===
        abs_img_path = fix_path(rel_img_path, root_dir)


        image = cv2.imread(abs_img_path)
        if image is None:
            return None, 0, "Cannot load image"

        x1, y1 = orig_box[0]
        x2, y2 = orig_box[1]
        H, W = image.shape[:2]

        nx1, ny1, nx2, ny2 = expand_box_with_min_area(x1, y1, x2, y2, enlarge_ratio, W, H, min_area=min_crop_size**2)
        crop_img = image[ny1:ny2, nx1:nx2]
        offset = (nx1, ny1)

        if crop_img.size == 0:
            return None, 0, "Empty crop"

        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        _, image_tensor = preprocess_pil_image(pil_crop)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor.squeeze(0),
                caption=question,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

        if boxes.shape[0] == 0:
            return None, 0, "No box detected"

        best_idx = int(np.argmax(logits.cpu().numpy()))
        if best_idx >= boxes.shape[0]:
            return None, 0, f"Selected index {best_idx} out of range"

        best_box = boxes[best_idx]
        crop_h, crop_w = crop_img.shape[:2]

        best_box_abs = adjust_box_to_original(best_box.cpu().numpy(), offset, (crop_w, crop_h))

        pred_box_tensor = torch.tensor([[best_box_abs[0][0], best_box_abs[0][1],
                                         best_box_abs[1][0], best_box_abs[1][1]]], dtype=torch.float)
        orig_box_tensor = xyxy_to_tensor(orig_box)
        iou = box_iou(orig_box_tensor, pred_box_tensor)[0][0].item()

        return best_box_abs, iou, None

    except torch.cuda.OutOfMemoryError as oom_err:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return None, 0, f"CUDA OOM: {str(oom_err)}"

    except Exception as e:
        torch.cuda.empty_cache()
        return None, 0, f"Exception: {str(e)}"

