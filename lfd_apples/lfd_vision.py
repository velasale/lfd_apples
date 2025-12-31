import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
# from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import pandas as pd


import os
import cv2
import numpy as np

from tqdm import tqdm

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dino_patch_heatmap_video(image_folder, output_video, frame_rate=30):

    """Generate a video overlaying DINOv2 patch-level feature heatmaps on images.
    Args:
        image_folder (str): Path to folder containing input images.
        output_video (str): Path to save the output video.
        frame_rate (int): Frame rate of the output video.
    """

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


def yolo_centers_video(
        image_folder, 
        output_video, 
        model_name='yolov8n-seg.pt', 
        frame_rate=30, 
        conf_thresh=0.4, 
        cross_size=10,
        font_scale=0.5):
    """
    Generate a video overlaying YOLO segmentation centers as yellow crosshairs
    and display the center coordinates next to each crosshair.
    """
    model = YOLO(model_name)

    # Prepare video writer
    sample_img_path = next((os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder))
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
    if sample_img_path is None:
        raise RuntimeError("No images found in the folder!")

    sample_img = cv2.imread(sample_img_path)
    H, W, _ = sample_img.shape
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (W, H))

    for fname in tqdm(sorted(os.listdir(image_folder))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)

        # Run YOLO inference
        results = model(img, conf=conf_thresh)

        for result in results:
            masks = result.masks
            if masks is not None:
                mask_data = masks.data.cpu().numpy()
                for mask in mask_data:
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    M = cv2.moments(mask_resized)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Draw yellow crosshair
                        color = (0, 255, 255)
                        cv2.line(img, (cX - cross_size, cY), (cX + cross_size, cY), color, 2)
                        cv2.line(img, (cX, cY - cross_size), (cX, cY + cross_size), color, 2)

                        # Display coordinates next to crosshair
                        coord_text = f"({cX},{cY})"
                        cv2.putText(img, coord_text, (cX + cross_size + 2, cY - cross_size - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        # Overlay filename at top-left
        cv2.putText(img, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1)

        video_writer.write(img)

    video_writer.release()
    print(f"YOLO crosshair video with coordinates saved to {output_video}")


def yolo_latent_heatmap_video(image_folder,
                                      output_video,
                                      model_name='yolov8n-seg.pt',
                                      frame_rate=30,
                                      layer_index=12,   # first backbone block
                                      imgsz=640,
                                      print_layer_shape=True):
    """
    Generate a video overlaying YOLOv8 latent feature heatmaps using model.predict().
    Handles preprocessing internally to avoid tensor mismatch errors.
    """

    model = YOLO(model_name)
    backbone_feature = {}

    # Forward hook: capture the output of a backbone layer
    def hook(module, input, output):
        backbone_feature['feat'] = output
        if print_layer_shape:
            print(f"Layer {layer_index} ({module.__class__.__name__}) output shape: {output.shape}")
        

    model.model.model[layer_index].register_forward_hook(hook)

    # Video writer setup
    sample_img_path = next((os.path.join(image_folder, f)
                            for f in sorted(os.listdir(image_folder))
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
    if sample_img_path is None:
        raise RuntimeError("No images found in folder!")
        

    sample_img = cv2.imread(sample_img_path)
    H, W, _ = sample_img.shape
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (W, H))

    # Process each image
    for fname in tqdm(sorted(os.listdir(image_folder))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, fname)
        img_cv = cv2.imread(img_path)

        # Forward pass: YOLO handles resizing/padding internally
        with torch.no_grad():
            _ = model.predict(source=img_cv, imgsz=imgsz, verbose=False)

        # Get latent feature map from the hook
        feat_map = backbone_feature['feat']

        # Convert to 2D heatmap (average over channels)
        heatmap = feat_map.squeeze(0).mean(0).cpu().numpy()
        heatmap = ((heatmap - heatmap.min()) / np.ptp(heatmap) * 255).astype(np.uint8)


        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_VIRIDIS)

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        cv2.putText(overlay, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        video_writer.write(overlay)

    video_writer.release()
    print(f"YOLO latent heatmap video saved to {output_video}")


def extract_pooled_latent_vector(
        img_cv,
        model,
        layer_index=12
):
    """
    Takes a single OpenCV image and returns a pooled latent vector.
    """
    backbone_feature = {}

    # hook used only for this call
    def hook(module, input, output):
        backbone_feature['feat'] = output

    # register hook
    handle = model.model.model[layer_index].register_forward_hook(hook)

    with torch.no_grad():
        _ = model.predict(source=img_cv, imgsz=640, verbose=False)

    # remove hook
    handle.remove()

    # get feature map
    feat_map = backbone_feature['feat']  # [1, C, H', W']
    feat_map = feat_map.squeeze(0)       # [C, H', W']

    # pooled latent vector
    pooled_vector = feat_map.mean(dim=(1, 2))  # [C]

    return pooled_vector.cpu().numpy(), feat_map


def pooled_latent_heatmap_video(
    image_folder, 
    output_video, 
    model_name, 
    layer_index=12, 
    frame_rate=30
):
    model = YOLO(model_name)

    # Get sample frame for video size
    sample_img_path = next(
        (os.path.join(image_folder, f)
         for f in sorted(os.listdir(image_folder))
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))),
        None
    )
    if sample_img_path is None:
        raise RuntimeError("No images found in folder!")

    sample_img = cv2.imread(sample_img_path)
    H, W, _ = sample_img.shape

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (W*2, H))  # double width for 2 images

    pooled_vectors = []

    for fname in tqdm(sorted(os.listdir(image_folder))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, fname)
        img_cv = cv2.imread(img_path)

        pooled_vector, feat_map = extract_pooled_latent_vector(
            img_cv, model, layer_index=layer_index
        )
        pooled_vectors.append(pooled_vector)

        feat_map_cpu = feat_map.detach().cpu().numpy()

        # Take first channel for visualization
        if feat_map_cpu.ndim == 3:
            heatmap = feat_map_cpu[0, :, :]
        elif feat_map_cpu.ndim == 4:
            heatmap = feat_map_cpu[0, 0, :, :]

        # Normalize to 0-255 and convert to uint8
        heatmap_norm = ((heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-6) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(cv2.resize(heatmap_norm, (W, H)), cv2.COLORMAP_VIRIDIS)

        # Side-by-side: left = raw image, right = heatmap
        side_by_side = np.concatenate([img_cv, heatmap_color], axis=1)

        # Optionally overlay filename
        cv2.putText(
            side_by_side, fname, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )

        video_writer.write(side_by_side)

    video_writer.release()
    print(f"Pooled latent heatmap video saved to {output_video}")

    # Save pooled vectors
    df = pd.DataFrame(pooled_vectors)
    df.to_csv("pooled_features.csv", index=False)
    print(df.shape[1], "features saved to pooled_features.csv")


def visualize_specific_channel(channel_number=23):

    # Load YOLO model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
    cv_model = YOLO(pt_path)
    cv_model.eval()

    layer15_out = None

    def hook_fn(module, input, output):
        nonlocal layer15_out   # 🔑 THIS IS THE FIX
        layer15_out = output.detach()

    cv_model.model.model[15].register_forward_hook(hook_fn)

    folder = '/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)/trial_1/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw'
    file = 'frame_00000_0.082738.jpg'
    filepath = os.path.join(folder, file)

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 384))
    img = cv2.rotate(img, cv2.ROTATE_180)

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        _ = cv_model.model(img_tensor)

    print(layer15_out.shape)  # ✅ now works

    channel = layer15_out[0, channel_number].cpu().numpy()  # 48 x 80

    # Resize the original image
    img_resized = cv2.resize(img, (640, 384))

    # Prepare heatmap
    heatmap = cv2.resize(channel, (640, 384))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw channel 21 activation
    im0 = axes[0].imshow(channel, cmap="inferno")
    axes[0].set_title(f"YOLO Layer 15 – Channel {channel_number}")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Activation")

    # Right: overlay on RGB image
    axes[1].imshow(img_resized)
    axes[1].imshow(heatmap, cmap="inferno", alpha=0.5)
    axes[1].set_title(f"Channel {channel_number} Activation Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



def main():
  
    trial = 'trial_1'
    for layer in range(22):
       
        frame_rate = 30  # frames per second
        
        image_path = '/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)/' + trial + '/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw'       
        yolo_output_video = "/home/alejo/Documents/temporal/" + trial + "_yolo_detection_video_" + str(layer) + "layers.mp4"
        
        # dino_patch_heatmap_video(image_folder, dino_output_video, frame_rate)
        # Get the path to the folder containing this script

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
    
        pooled_latent_heatmap_video(image_path, yolo_output_video, model_name=pt_path, layer_index=layer, frame_rate=frame_rate)

if __name__ == "__main__":

    # main()  

    visualize_specific_channel()
    