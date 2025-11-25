# ---------------------------
# Settings
# ---------------------------
image_folder = "/home/alejo/Documents/temporal/trial_5/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw"
output_video = "/home/alejo/Documents/temporal/trial_5/dinov2_patch_heatmap_video.mp4"
frame_rate = 30  # frames per second

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ---------------------------
# Load DINOv2 model
# ---------------------------
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
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
    """Extract patch-level features from DINOv2"""
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(pixel_values=img_t)
        feats = outputs.last_hidden_state  # (1, num_tokens, hidden_dim)
        patch_feats = feats[:, 1:, :]      # remove CLS token

    return patch_feats.squeeze(0).cpu().numpy(), img

# ---------------------------
# Prepare video writer
# ---------------------------
# Get a sample image to determine frame size
sample_img_path = next((os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder))
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
if sample_img_path is None:
    raise RuntimeError("No images found in the folder!")

sample_img = Image.open(sample_img_path).convert("RGB")
W, H = sample_img.size
W, H = int(W), int(H)  # make sure they are integers

# Ensure output directory exists
os.makedirs(os.path.dirname(output_video), exist_ok=True)

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

    # Normalize heatmap 0-255 (NumPy 2.x compatible)
    heatmap_resized = ((heatmap_resized - heatmap_resized.min()) / np.ptp(heatmap_resized) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_VIRIDIS)

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    # Write frame
    video_writer.write(overlay)

video_writer.release()
print(f"Video saved to {output_video}")
