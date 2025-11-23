import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os
from tqdm import tqdm

# ---------------------------
# Settings
# ---------------------------
image_folder = "/home/guest/My Projects/Temporal/trial_103/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw"
output_video = "/home/guest/My Projects/Temporal/trial_103/trial_103_dino_heatmap_video.mp4"
frame_rate = 5  # frames per second

# ---------------------------
# Load DINO model
# ---------------------------
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
model.eval()

# ---------------------------
# Preprocessing
# ---------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def extract_patch_features(img_path):
    """Extract patch features (excluding CLS token)"""
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        feats = model.get_intermediate_layers(img_t, n=1)[0]  # shape: (1, 197, 384)
        patch_feats = feats[:, 1:, :]  # remove CLS token

    return patch_feats.squeeze(0).cpu().numpy(), img

# ---------------------------
# Prepare video writer
# ---------------------------
sample_img = Image.open(os.path.join(image_folder, sorted(os.listdir(image_folder))[0])).convert("RGB")
W, H = sample_img.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (W, H))

# ---------------------------
# Process each image
# ---------------------------
for fname in tqdm(sorted(os.listdir(image_folder))):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(image_folder, fname)
    patch_feats, img = extract_patch_features(img_path)

    # Average over embedding dim to get single value per patch
    patch_map = patch_feats.mean(axis=1)
    grid_size = int(np.sqrt(patch_map.shape[0]))
    patch_map = patch_map.reshape(grid_size, grid_size)

    # Upsample heatmap to original image size
    heatmap_resized = np.array(Image.fromarray(patch_map).resize((W, H), resample=Image.BILINEAR))

    # Normalize heatmap 0-255
    heatmap_resized = ((heatmap_resized - heatmap_resized.min()) / (heatmap_resized.ptp()) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_VIRIDIS)

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    # Write frame
    video_writer.write(overlay)

video_writer.release()
print(f"Video saved to {output_video}")
