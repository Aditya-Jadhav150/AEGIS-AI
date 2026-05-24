import cv2
import numpy as np
import math
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

class GeometricAligner:
    """
    Module 1: Preprocessing & Facial Landmark Alignment Pipeline
    1. Extracts 5 facial landmarks using MTCNN (facenet_pytorch).
    2. Computes the orientation angle and executes an affine transformation to align eyes horizontally.
    3. Crops around the center of mass with a 10% outer padding margin.
    4. Resizes to 512x512 using bi-cubic interpolation and normalizes.
    """
    def __init__(self, device='cpu'):
        if MTCNN is None:
            raise ImportError("facenet_pytorch is required. Install via 'pip install facenet-pytorch'")
            
        # Initialize MTCNN for face and landmark detection
        self.device = device
        self.mtcnn = MTCNN(keep_all=False, device=self.device, margin=0, post_process=False)
        
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def align_and_crop(self, image_bgr: np.ndarray, return_tensor=True):
        """
        Executes the alignment and cropping pipeline on a BGR numpy image (e.g. from cv2.imread).
        If return_tensor=True, returns a normalized torch Tensor of shape [3, 512, 512].
        If return_tensor=False, returns an RGB numpy array of shape [512, 512, 3].
        Returns None if no face is detected.
        """
        # MTCNN works best with RGB PIL Images or numpy arrays
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None
        
        # Take the most prominent face (MTCNN keep_all=False returns the one with highest probability by default)
        # But detect returns arrays, so we take the first index
        bbox = boxes[0]
        pts = landmarks[0] # Shape: (5, 2)
        
        # 1. 5 primary facial landmarks
        # MTCNN landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = pts[0]
        right_eye = pts[1]
        
        # 2. Compute orientation angle relative to the base plane
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Determine center of mass of landmarks for rotation center
        center_of_mass = np.mean(pts, axis=0)
        center_x, center_y = int(center_of_mass[0]), int(center_of_mass[1])
        
        # Execute affine transformation matrix mapped around center of mass
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        h, w = image_bgr.shape[:2]
        rotated_img = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Re-detect on rotated image ensures accurate bounding box after rotation
        rotated_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
        r_boxes, r_probs, r_landmarks = self.mtcnn.detect(Image.fromarray(rotated_rgb), landmarks=True)
        
        if r_boxes is None or len(r_boxes) == 0:
            # Fallback to rotating the bounding box manually
            pts_box = np.array([
                [bbox[0], bbox[1], 1],
                [bbox[2], bbox[1], 1],
                [bbox[2], bbox[3], 1],
                [bbox[0], bbox[3], 1]
            ])
            pts_rot = M.dot(pts_box.T).T
            min_x, min_y = np.min(pts_rot[:, 0]), np.min(pts_rot[:, 1])
            max_x, max_y = np.max(pts_rot[:, 0]), np.max(pts_rot[:, 1])
            rotated_bbox = [min_x, min_y, max_x, max_y]
        else:
            # Use re-detected bounding box (most accurate)
            rotated_bbox = r_boxes[0]
            
        # 3. Isolate the cropping window boundaries with 10% outer padding margin
        bx1, by1, bx2, by2 = rotated_bbox
        bw = bx2 - bx1
        bh = by2 - by1
        
        pad_w = bw * 0.10
        pad_h = bh * 0.10
        
        cx = bx1 + bw / 2
        cy = by1 + bh / 2
        
        # Make the crop square to avoid distortion during 512x512 resize
        side = max(bw + 2 * pad_w, bh + 2 * pad_h)
        
        nx1 = int(max(0, cx - side / 2))
        ny1 = int(max(0, cy - side / 2))
        nx2 = int(min(w, cx + side / 2))
        ny2 = int(min(h, cy + side / 2))
        
        crop_img = rotated_img[ny1:ny2, nx1:nx2]
        
        # 4. Bi-cubic resizing to 512x512
        resized_img = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        
        # Convert BGR to RGB
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        if not return_tensor:
            return resized_img_rgb
            
        # Convert to Tensor, normalize to [0, 1] then ImageNet bounds
        tensor_img = torch.from_numpy(resized_img_rgb).permute(2, 0, 1).float() / 255.0
        normalized_img = self.normalize(tensor_img)
        
        return normalized_img

