import cv2
import numpy as np
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# Initialize BLIP model for caption generation
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
# Function to query the Mistral model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer add your token"}

# Function to query the Mistral model
def query_mistral(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    response_data = response.json()
    script = ""
    if isinstance(response_data, list) and response_data:
        script = response_data[0].get("generated_text", "")
    elif "generated_text" in response_data:
        script = response_data.get("generated_text", "")
    
    if script:
        script = script.rsplit('.', 1)[0] + '.'  # Ensure it ends with a complete sentence
    return script


# Step 1: User Uploads Video
root = tk.Tk()
root.withdraw()  # Hide the main window
video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.flv;*.mov")])
if not video_path:
    print("No video selected.")
    exit()

# Step 2: Detect Scene Changes and Save Frames
cap = cv2.VideoCapture(video_path)


prev_frame = None
threshold = 13000000 #Adjust this according to your video
prev_scene_time = 0
min_scene_duration = 3000  #Adjust this according to your video
image_count = 0
saved_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if prev_frame is None:
        prev_frame = frame
        continue
    diff = cv2.absdiff(frame, prev_frame)
    diff_sum = np.sum(diff)
    if diff_sum > threshold:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_time - prev_scene_time > min_scene_duration:
            image_name = f"scene_change_{image_count}.jpg"
            cv2.imwrite(image_name, frame)
            saved_frames.append(image_name)
            image_count += 1
            prev_scene_time = current_time
    prev_frame = frame

cap.release()


# Step 3: Generate Voiceover Scripts for Saved Frames
captions = []
voiceover_scripts = []
for frame_path in saved_frames:
    frame_image = Image.open(frame_path)
    inputs = processor(frame_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append(caption)

    # Generate voiceover script using the caption
    script = query_mistral(caption)
    voiceover_scripts.append(script)


# Create storyboard PDF with voiceover scripts
pdf_path = "storyboard.pdf"
with PdfPages(pdf_path) as pdf:
    for i in range(0, len(saved_frames), 6):
        fig, axes = plt.subplots(2, 3, figsize=(16.53, 11.69))
        fig.tight_layout(pad=3.0)
        print(fig.tight_layout(pad=3.0))
        
        # Adjust for the actual number of frames in the last iteration if it's less than 6
        num_frames = min(len(saved_frames) - i, 6)
        
        for j in range(num_frames):
            frame_path = saved_frames[i+j]
            script = voiceover_scripts[i+j]
            
            ax = axes.flat[j]
            img = plt.imread(frame_path)
            ax.imshow(img)
            script_wrapped = '\n'.join(textwrap.wrap(script, width=60))
            ax.set_title(script_wrapped, size=10, pad=20)

            ax.axis('off')
        
        pdf.savefig(fig)
        plt.close(fig)

print(f"Storyboard PDF created: {pdf_path}")
