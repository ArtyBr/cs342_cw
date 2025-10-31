import torch
import timm
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# 1. Load pretrained Vision Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit.eval() # disable dropout, etc.

# 2. Define preprocessing consistent with ImageNet training
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
),
])

# 3. Directory containing CelebA images (modify path)
image_dir = "/home/arty/projects/cs342_dataset/img_align_celeba"
image_list = sorted(os.listdir(image_dir))[:20000] # sample subset

# Get labeled attributes from dataset
attr_file = "/home/arty/projects/cs342_dataset/list_attr_celeba.csv"
attr_df = pd.read_csv(attr_file)

# Split images into train (70%). calib (15%), test (15%)
train_set, remaining_set = train_test_split(
    image_list, 
    test_size=0.30, 
    random_state=42, # Use a fixed random_state for reproducibility
    shuffle=True
)
calib_set, test_set = train_test_split(
    remaining_set, 
    test_size=0.50, # 50% of the remaining 30% = 15% of the total
    random_state=42, # Using the same random_state is good practice
    shuffle=True
)

sets = [train_set, calib_set, test_set]

# 4. Extract embeddings
for split_name, image_list in zip(["train", "calib", "test"], sets):
    print(f"Processing {split_name} set with {len(image_list)} images...")

    all_features = []
    with torch.no_grad():
        for fname in tqdm(image_list):
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            features = vit.forward_features(x)[:, 0, :] # shape: (1, 768)
            all_features.append(features.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)

    # Add 'Smiling' class label as the last column
    smiling_labels = []
    for fname in image_list:
        index = int(fname.split('.')[0]) 
        smiling_label = 1 if attr_df.at[index, 'Smiling'] == 1 else 0
        smiling_labels.append(smiling_label)
    smiling_labels = np.array(smiling_labels).reshape(-1, 1)
    all_features = np.hstack((all_features, smiling_labels))

    print("Feature matrix shape:", all_features.shape) # e.g. (20000, 768 + 1)

    # 5. Save for later use
    np.save(f"celeba_vit_embeddings_{split_name}.npy", all_features)