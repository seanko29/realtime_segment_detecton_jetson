import os
import warnings
import gradio as gr
import numpy as np
from PIL import Image
from llm import efficientViT_SAM  
from webcam import *
from ultralytics import YOLO
import time

import torch

# Set environment variables for CUDA
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_ROOT"] = "/usr/local/cuda"
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


warnings.filterwarnings("ignore")
sam_model = efficientViT_SAM()  # Example model; adjust as needed
realtime_model = Realtime_Lang_Sam(sam_model)
yolo_world_m = YOLO('yolov8x-worldv2.pt')


def predict(conf_threshold, iou_threshold, input, text_prompt):
    text_prompt = text_prompt.split(',')
    
    # Force model to GPU before each prediction
    yolo_world_m.to('cuda')
    
    # Clear CUDA cache to prevent device conflicts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure inference mode
    with torch.inference_mode():  # or torch.no_grad()
        realtime_model.init_model_medium(prompt=text_prompt, model=yolo_world_m)
        output = realtime_model.predict_frame(input, conf_threshold, iou_threshold)
    
    return output 

def video_predict(conf_threshold, iou_threshold, input, text_prompt):
    text_prompt = text_prompt.split(',')
    
    # Your existing model code...
    yolo_world_m.to('cuda')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    realtime_model.init_model_medium(prompt=text_prompt, model=yolo_world_m)
    output_path = realtime_model.predict_video(input, conf_threshold, iou_threshold)
    
    if output_path and os.path.exists(output_path):
        # Return the video file path directly
        # Gradio will handle the file reading
        return output_path
    
    return None
# def video_predict(conf_threshold, iou_threshold, input, text_prompt):
#     text_prompt = text_prompt.split(',')
    
#     # Force model to GPU before each prediction
#     yolo_world_m.to('cuda')
    
#     # Clear CUDA cache to prevent device conflicts
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     # Ensure inference mode
#     with torch.inference_mode():  # or torch.no_grad()
#         realtime_model.init_model_medium(prompt=text_prompt, model=yolo_world_m)
#         output = realtime_model.predict_video(input, conf_threshold, iou_threshold)
#         # output = realtime_model.predict_video_imageio(input, conf_threshold, iou_threshold)
    
#     return output

def realtime_predict(prompt):
    prompt = prompt.split(',')
    realtime_model.init_model(prompt)
    realtime_model.predict_realtime()


# Define the input components directly from the 'gr' namespace
inputs = [
    gr.Slider(minimum=0, maximum=1, value=0.3, label="CONF threshold"),
    gr.Slider(minimum=0, maximum=1, value=0.25, label="IOU threshold"),
    gr.Image(label='Image', type='pil'),  # Now expecting a PIL Image directly
    gr.Textbox(label="Text Prompt", lines=2, placeholder="Enter text here..."),
]

inputs2 = [
    gr.Slider(minimum=0, maximum=1, value=0.3, label="CONF threshold"),
    gr.Slider(minimum=0, maximum=1, value=0.25, label="IOU threshold"),
    gr.Video(),
    gr.Textbox(label="Text Prompt", lines=2, placeholder="Enter text here..."),
]


# Define the output component directly from the 'gr' namespace
output = gr.Image(label="Output Image")
output2 = gr.Video(label="Output Video")
# Example data
examples = [
    [0.36, 0.25, "assets/fig/cat.jpg", "cat"],
    [0.36, 0.25, "assets/demo/fruits.jpg", "bowl"],
]


with gr.Blocks() as app3:
    gr.Markdown(
    """
    # Language-Segment-Anything Realtime!
    """)
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Type your prompt here...")
        start_button = gr.Button("Start Processing")



    # Define the action to take when the start button is clicked
    start_button.click(fn=realtime_predict, inputs=prompt_input, concurrency_id="fn")

# Create the interface
app1 = gr.Interface(fn=predict, inputs=inputs, outputs=output, examples=examples, title="Language-Segment-Anything Photo Prediction", description="Generates predictions using the LangSAM model.")

app2 = gr.Interface(fn=video_predict,inputs=inputs2,outputs=output2,title="Language-Segment-Anything Video Prediction")
# Launch the interface
demo = gr.TabbedInterface([app1, app2, app3], ["Image", "Video", "Webcam"])
demo.queue(default_concurrency_limit=1)
# Try these launch parameters:
demo.launch(
    server_name="0.0.0.0",  # Keep this for listening on all interfaces
    server_port=7860,
    share=True,
    debug=True,
    show_error=True,
    root_path="/",
    allowed_paths=["/"]
)