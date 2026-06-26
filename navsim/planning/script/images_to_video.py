import cv2
import os
from pathlib import Path
from tqdm import tqdm

def images_to_video(input_dir, output_path, fps=10):
    """
    input_dir: Path to folder containing images (e.g.: /workspace/.../log_name)
    output_path: Output video file path (e.g.: scene_video.mp4)
    fps: Frames per second (default 10, higher value = faster playback)
    """
    image_folder = Path(input_dir)
    # Get and sort image file list (by filename)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

    if not images:
        print(f"No images found in folder: {input_dir}")
        return

    # Read the first image to determine video resolution (width, height)
    first_image_path = str(image_folder / images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Video codec setting (mp4v is for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting video creation: {output_path}")
    print(f"Total frames: {len(images)} | Resolution: {width}x{height} | FPS: {fps}")

    for image in tqdm(images):
        image_path = str(image_folder / image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video creation complete!")

# --- Usage example ---
if __name__ == "__main__":
    # Enter the path to a specific log folder inside the previously configured OUTPUT_DIR.
    target_scene_dir = "debug_viz2/2021.05.25.14.16.10_veh-35_01690_02183"
    log = os.path.basename(target_scene_dir)
    output_file = f"{target_scene_dir}.mp4"
    
    images_to_video(target_scene_dir, output_file, fps=5)