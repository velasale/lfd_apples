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


def yolo_detection_video(image_folder, output_video, model_name='yolov8n.pt', frame_rate=30):
    """
    Generate a video overlaying YOLO detections on images.
    
    Args:
        image_folder (str): Folder with input images.
        output_video (str): Path to save the output video.
        model_name (str): YOLO model checkpoint ('yolov8n.pt', 'yolov8s.pt', etc.)
        frame_rate (int): FPS of output video.
    """
    # Load YOLO model
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

    # Process images
    for fname in tqdm(sorted(os.listdir(image_folder))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)

        # Run YOLO inference
        results = model(img)

        # Draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids):
                label = f"{model.names[class_id]} {score:.2f}"
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(img, label, (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        video_writer.write(img)

    video_writer.release()
    print(f"YOLO video saved to {output_video}")


def main():
  
    image_folder = "/home/alejo/Documents/temporal/trial_5/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw"
    output_video = "/home/alejo/Documents/temporal/trial_5/dinov2_patch_heatmap_video.mp4"
    frame_rate = 30  # frames per second

    dino_patch_heatmap_video(image_folder, output_video, frame_rate)


if __name__ == "__main__":
    main()  
    