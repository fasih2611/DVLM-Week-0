import torchvision
import torch
import clip
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from torch.utils.data import DataLoader
from tqdm import tqdm


data_test = torchvision.datasets.STL10("./dataset",split="test",folds=0,download="True")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def part_1():
    num_classes = len(data_test.classes)
    prompts = {"plain":["" for _ in range(num_classes)],"prompted":["a photo of " for _ in range(num_classes)],
               "descriptive":["a photo of a airplane in the sky","a photo of a bird flying","a photo of a car on the road",
                              "a photo of a cat looking at the camera","a photo of a deer in the wild","a photo of a dog looking at the camera",
                              "a photo of a horse doing horse things","a photo of a monkey in its habitat","a photo of a ship in the sea",
                              "a photo of a truck on the street"]}
    for key in prompts.keys():
        for class_id in range(num_classes):
            if key != "descriptive":
                prompts[key][class_id] += data_test.classes[class_id]

    prompt_accuracy = {"plain":0.0,"prompted":0.0,"descriptive":0.0}
    for image,label in data_test:
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            for stratagy in prompt_accuracy.keys():
                text = clip.tokenize(prompts[stratagy]).to(device)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                class_pred = probs.argmax()
                if class_pred == label:
                    prompt_accuracy[stratagy] += 1
    total_images = len(data_test)
    for strategy in prompt_accuracy.keys():
        prompt_accuracy[strategy] = (prompt_accuracy[strategy] / total_images) * 100
        print(f"{strategy}: {prompt_accuracy[strategy]:.4f}")

def part_2():
    num_samples = 100
    np.random.seed(42)
    indices = np.random.choice(len(data_test), num_samples, replace=False)
    image_embeddings = []
    labels = []
    text_prompts = [f"a photo of a {class_name}" for class_name in data_test.classes]
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_embeddings_all = text_features.cpu().numpy()
        for idx in indices:
            image, label = data_test[idx]
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            image_embeddings.append(image_features.cpu().numpy())
            labels.append(label)
    image_embeddings = np.vstack(image_embeddings)
    labels = np.array(labels)
    text_embeddings_sampled = text_embeddings_all[labels]
    
    all_embeddings = np.vstack([image_embeddings, text_embeddings_sampled])
    reducer_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d_umap = reducer_umap.fit_transform(all_embeddings)
    print("GENERATED")

    image_2d_umap = embeddings_2d_umap[:num_samples]
    text_2d_umap = embeddings_2d_umap[num_samples:]

    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings_sampled / np.linalg.norm(text_embeddings_sampled, axis=1, keepdims=True)
    
    all_embeddings_norm = np.vstack([image_embeddings_norm, text_embeddings_norm])
    reducer_umap_norm = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d_norm_umap = reducer_umap_norm.fit_transform(all_embeddings_norm)
    
    image_2d_norm = embeddings_2d_norm_umap[:num_samples]
    text_2d_norm = embeddings_2d_norm_umap[num_samples:]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ax = axes[0, 0]
    scatter1 = ax.scatter(image_2d_umap[:, 0], image_2d_umap[:, 1], c=labels, cmap='tab10', 
                          alpha=0.6, s=80, label='images', marker='o', edgecolors='black', linewidths=0.5)
    scatter2 = ax.scatter(text_2d_umap[:, 0], text_2d_umap[:, 1], c=labels, cmap='tab10', 
                          alpha=0.9, s=120, label='Text', marker='X', edgecolors='black', linewidths=1.5)
    ax.set_title('UMAP: image and text embeddings by class', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.scatter(image_2d_umap[:, 0], image_2d_umap[:, 1], c='blue', 
               alpha=0.7, s=80, label='Images', edgecolors='black', linewidths=0.5)
    ax.scatter(text_2d_umap[:, 0], text_2d_umap[:, 1], c='red', 
               alpha=0.7, s=80, label='Text', edgecolors='black', linewidths=0.5)
    ax.set_title('UMAP: Modality Separation (Image vs Text)', fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(image_2d_norm[:, 0], image_2d_norm[:, 1], c=labels, cmap='tab10', 
               alpha=0.6, s=80, label='Images (normalized)', marker='o', edgecolors='black', linewidths=0.5)
    ax.scatter(text_2d_norm[:, 0], text_2d_norm[:, 1], c=labels, cmap='tab10', 
               alpha=0.9, s=120, label='Text (normalized)', marker='X', edgecolors='black', linewidths=1.5)
    ax.set_title('UMAP: normalized embeddings', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    distances_unnorm = np.linalg.norm(image_embeddings - text_embeddings_sampled, axis=1)
    distances_norm = np.linalg.norm(image_embeddings_norm - text_embeddings_norm, axis=1)
    ax.hist(distances_unnorm, bins=25, alpha=0.7, label='unnormalized', color='blue', edgecolor='black')
    ax.hist(distances_norm, bins=25, alpha=0.7, label='normalized', color='red', edgecolor='black')
    ax.set_title('euclidean distance Distribution\n matching image-text pair', fontsize=13, fontweight='bold')
    ax.set_xlabel('distance', fontsize=11)
    ax.set_ylabel('frequency', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 2]
    cosine_similarities = np.sum(image_embeddings_norm * text_embeddings_norm, axis=1)
    ax.hist(cosine_similarities, bins=30, alpha=0.8, color='green', edgecolor='black')
    ax.axvline(x=np.mean(cosine_similarities), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(cosine_similarities):.3f}')
    ax.set_title('cosine similarity distribution\n matching Image-Text pair', fontsize=13, fontweight='bold')
    ax.set_xlabel('cosine similarity', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('./clip embedding visualization.png', dpi=300, bbox_inches='tight')
    
    print(f"Average Euclidean distance unnormalized:{np.mean(distances_unnorm):.4f} ± {np.std(distances_unnorm):.4f}")
    print(f"Average Euclidean distance normalized:{np.mean(distances_norm):.4f} ± {np.std(distances_norm):.4f}")
    print(f"Average cosine similarity:{np.mean(cosine_similarities):.4f} ± {np.std(cosine_similarities):.4f}")
    print(f"Min cosine similarity:{np.min(cosine_similarities):.4f}")
    print(f"Max cosine similarity{np.max(cosine_similarities):.4f}")
    
    print(f"Distance reduction: {np.mean(distances_unnorm):.3f} {np.mean(distances_norm):.3f}")
    
    n_random = min(500, num_samples * 9)
    random_pairs = []
    for i in range(num_samples):
        non_matching = [j for j in range(len(text_embeddings_all)) if j != labels[i]]
        sampled = np.random.choice(non_matching, min(9, len(non_matching)), replace=False)
        for j in sampled:
            img_norm = image_embeddings_norm[i]
            txt_norm = text_embeddings_all[j] / np.linalg.norm(text_embeddings_all[j])
            cos_sim = np.dot(img_norm, txt_norm)
            random_pairs.append(cos_sim)
    
    random_pairs = np.array(random_pairs[:n_random])
    
    print(f"Matching pairs (image-correct text):{np.mean(cosine_similarities):.4f} ± {np.std(cosine_similarities):.4f}")
    print(f"Non-matching pairs (image-wrong text):{np.mean(random_pairs):.4f} ± {np.std(random_pairs):.4f}")
    print(f"Separation margin:{np.mean(cosine_similarities) - np.mean(random_pairs):.4f}")

def part_3():
    classes = data_test.classes

    def get_all_features(dataset, model, batch_size=32):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        all_image_features = []
        all_text_features = []
        all_labels = []
        print("Extracting features...")
        with torch.no_grad():
            # This was taking too long on a cpu so added tqdm
            for images, labels in tqdm(loader):
                images = images.to(device)
                img_feats = model.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True) # Normalize
                all_image_features.append(img_feats.cpu().numpy())
                
                batch_prompts = [f"a photo of a {classes[label]}" for label in labels]
                text_tokens = clip.tokenize(batch_prompts).to(device)
                txt_feats = model.encode_text(text_tokens)
                txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True) 
                all_text_features.append(txt_feats.cpu().numpy())
                all_labels.append(labels.numpy())

        X = np.vstack(all_image_features) # Image Embeddings
        Y = np.vstack(all_text_features)  #  Text Embeddings
        labels = np.concatenate(all_labels)
        
        return X, Y, labels

    X, Y, labels = get_all_features(data_test, model)

    # embed generation for all the classes, used the caption for consistant results
    class_prompts = [f"a photo of a {c}" for c in classes]
    with torch.no_grad():
        class_tokens = clip.tokenize(class_prompts).to(device)
        class_emb = model.encode_text(class_tokens)
        class_emb = class_emb / class_emb.norm(dim=1, keepdim=True)
        W_text = class_emb.cpu().numpy() # (10, 512)

    logits = X @ W_text.T
    preds = np.argmax(logits, axis=1)
    acc_baseline = np.mean(preds == labels) * 100
    print(f"\nBaseline acc: {acc_baseline:.4f}")

    R, scale = orthogonal_procrustes(X, Y)

    print(f"Rotation Matrix R shape: {R.shape}")

    X_aligned = X @ R
    logits_aligned = X_aligned @ W_text.T
    preds_aligned = np.argmax(logits_aligned, axis=1)
    acc_aligned = np.mean(preds_aligned == labels) * 100
    print(f"Procrustes Accuracy: {acc_aligned:.2f}")
    print(f"Improvement: {acc_aligned - acc_baseline:.2f}")

    num_samples = 500
    indices = np.random.choice(len(X), num_samples, replace=False)

    X_sub = X[indices]
    Y_sub = Y[indices]
    labels_sub = labels[indices]

    X_aligned_sub = X_aligned[indices]

    reducer = umap.UMAP(n_components=2, random_state=42)

    all_vecs = np.vstack([X_sub, Y_sub, X_aligned_sub])
    embedding_2d = reducer.fit_transform(all_vecs)

    emb_img_orig = embedding_2d[:num_samples]
    emb_text= embedding_2d[num_samples:2*num_samples]
    emb_img_align= embedding_2d[2*num_samples:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(emb_img_orig[:, 0], emb_img_orig[:, 1], c='blue', alpha=0.5, label='Images (Original)', s=15)
    axes[0].scatter(emb_text[:, 0], emb_text[:, 1], c='red', alpha=0.5, label='Text', s=15)
    axes[0].set_title("Before Alignment")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(emb_img_align[:, 0], emb_img_align[:, 1], c='green', alpha=0.5, label='Images (Aligned)', s=15)
    axes[1].scatter(emb_text[:, 0], emb_text[:, 1], c='red', alpha=0.5, label='Text', s=15)
    axes[1].set_title("After Procrustes Alignment")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("procrustes_alignment.png")
    plt.show()


if __name__ == "__main__":
    # part_1()
    # part_2()
    part_3()