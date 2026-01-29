import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224").to(device)


def get_attention_map(img, model, processor, device,heads_to_use=None):
    inputs = processor(img, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions[-1]  # (batch, heads, seq_len, seq_len)
    if heads_to_use is None:
        cls_attention = attentions[0, :, 0, 1:].mean(dim=0) # Average over heads
    else:
        cls_attention = attentions[0, :heads_to_use, 0, 1:].mean(dim=0)
    # Reshape to 14x14 (since 224/16 = 14)
    att_map = cls_attention.view(14, 14).detach().cpu().numpy()
    att_map = cv2.resize(att_map, (img.width, img.height))
    return att_map, outputs.logits.argmax(-1).item()

def visualize_overlay(img, att_map, pred_label):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(att_map, cmap='jet', alpha=0.5) 
    plt.title(f"Pred: {model.config.id2label[pred_label].split(',')[0]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def mask_image(img_tensor, mask_type="random", ratio=0.25):
    # Masking logic for Task 4q
    b, c, h, w = img_tensor.shape
    patch_size = 16
    h_patches, w_patches = h // patch_size, w // patch_size
    masked = img_tensor.clone()
    
    if mask_type == "random":
        num_masked = int(h_patches * w_patches * ratio)
        indices = np.random.choice(h_patches * w_patches, num_masked, replace=False)
        for idx in indices:
            r, c = divmod(idx, w_patches)
            masked[:, :, r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = 0
            
    elif mask_type == "center":
        center_h, center_w = h_patches // 2, w_patches // 2
        extent = int(np.sqrt(h_patches * w_patches * ratio) / 2)
        r_s, r_e = max(0, center_h - extent), min(h_patches, center_h + extent)
        c_s, c_e = max(0, center_w - extent), min(w_patches, center_w + extent)
        masked[:, :, r_s*patch_size:r_e*patch_size, c_s*patch_size:c_e*patch_size] = 0
        
    return masked

def run_tasks():
    # local samples from imagenet within the git repo, didn't want to use relative path since idk where this script will be excuted from
    images = []
    images_dir = pathlib.Path(__file__).resolve().parent / 'images'
    for image in images_dir.glob("*.JPEG"):
        images.append(Image.open(image))

    for img in images:
        # set heads_to_use to None to use all heads
        att_map, pred = get_attention_map(img, model, processor, device,heads_to_use=1)
        print(f"Prediction: {model.config.id2label[pred]}")
        visualize_overlay(img, att_map, pred)
    inputs = processor(images[1], return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']
    for m_type in ["random", "center"]:
        masked_pv = mask_image(pixel_values, mask_type=m_type, ratio=0.4)
        with torch.no_grad():
            logits = model(pixel_values=masked_pv).logits
        pred_idx = logits.argmax(-1).item()
        print(f"Mask: {m_type: <8} | Prediction: {model.config.id2label[pred_idx].split(',')[0]}")

    ds = load_dataset("uoft-cs/cifar10", split="train[:1000]") 
    X_cls, X_mean, y = [], [], []
    for item in ds:
        inp = processor(item['img'], return_tensors="pt").to(device)
        with torch.no_grad():
            out = feature_extractor(**inp)
            # CLS is token 0, Mean is average of tokens 1-X
            X_cls.append(out.last_hidden_state[0, 0, :].cpu().numpy())
            X_mean.append(out.last_hidden_state[0, 1:, :].mean(dim=0).cpu().numpy())
            y.append(item['label'])
            
    X_cls, X_mean, y = np.array(X_cls), np.array(X_mean), np.array(y)
    split = int(0.8 * len(y))
    clf_cls = LogisticRegression(max_iter=500).fit(X_cls[:split], y[:split])
    clf_mean = LogisticRegression(max_iter=500).fit(X_mean[:split], y[:split])
    acc_cls = accuracy_score(y[split:], clf_cls.predict(X_cls[split:]))
    acc_mean = accuracy_score(y[split:], clf_mean.predict(X_mean[split:]))
    print(f"CLS Token Accuracy: {acc_cls:.4f}")
    print(f"Mean Pool Accuracy: {acc_mean:.4f}")

if __name__ == "__main__":
    run_tasks()