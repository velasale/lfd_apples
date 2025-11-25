import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm


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


def main():
  
    image_folder = "/home/alejo/Documents/temporal/trial_5/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw"
    # dino_output_video = "/home/alejo/Documents/temporal/trial_5/dinov2_patch_heatmap_video.mp4"
    yolo_output_video = "/home/alejo/Documents/temporal/trial_5/yolo_detection_video.mp4"
    frame_rate = 30  # frames per second

    # dino_patch_heatmap_video(image_folder, dino_output_video, frame_rate)

    # Get the path to the folder containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
    # yolo_centers_video(image_folder, yolo_output_video , model_name=pt_path, frame_rate=frame_rate)


    # Example usage
    yolo_output_video = "/home/alejo/Documents/temporal/trial_5/yolo_latent_video.mp4"
    yolo_latent_heatmap_video(image_folder, yolo_output_video, model_name=pt_path)


if __name__ == "__main__":
    main()  
    