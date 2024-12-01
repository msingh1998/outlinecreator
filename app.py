import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import io

# Initialize MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def create_outline_effect(image, outline_size=10):
    # Convert from Gradio's PIL image to CV2 format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    scale_factor = min(width, height) / 1000
    adjusted_size = max(3, int(outline_size * scale_factor))
    
    # Process with RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_image)
    mask = results.segmentation_mask
    
    if mask is None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Return original image in RGB
    
    # Rest of the processing remains the same until the end
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    kernel_size = max(3, adjusted_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    dilated = cv2.dilate(mask, kernel, iterations=1)
    dilated = cv2.GaussianBlur(dilated, (5, 5), 0)
    
    edge = cv2.subtract(dilated, mask)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    
    # Create white outline
    white_outline = np.zeros_like(image)
    white_outline[edge > 127] = [255, 255, 255]
    
    # Blend with original
    result = cv2.addWeighted(image, 1, white_outline, 1, 0)
    
    # Before returning, convert to PIL Image and save as PNG
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result)
    
    # Save to a BytesIO object as PNG
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG', quality=100)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Convert back to PIL Image
    final_image = Image.open(io.BytesIO(img_byte_arr))
    return final_image

def resize_if_needed(image, max_dimension=1200):
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Create Gradio interface
demo = gr.Interface(
    fn=create_outline_effect,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Outline Size")
    ],
    outputs=gr.Image(
        label="Result",
        type="numpy",
        format="png"
    ),
    title="Image Outline Effect Generator",
    description="Upload an image to add a white outline effect around the subject.",
    allow_flagging="never"
)

# Launch with file serving enabled
demo.launch(share=True)