from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from llm import draw_image, efficientViT_SAM
import time

cv2.startWindowThread()

class Realtime_Lang_Sam:
    def __init__(self, langSAM):
        self.sam = langSAM
        self.model = None
        self.names = None

    def init_model(self,prompt):
      
        if prompt is None:
            self.model = YOLO('custom.pt')
        else:
            print("Creating custom model .....")
            self.model = YOLO('yolov8s-world.pt')
            self.model.set_classes(prompt)
            self.model.save("custom.pt")
            self.model = YOLO('custom.pt')
            print("model loaded ....")

        self.names = self.model.names
        print("finish initializing model ....")
    
    def init_model_medium(self,prompt,model):
        self.model = model
        start = time.time()

        self.model.set_classes(prompt)
        end = time.time()
        print("Elapsed Time set_classes : " , (end - start)*1000 )

        self.names = self.model.names

    def predict_frame(self,unprocessed_image,conf,iou):
        image_array = np.asarray(unprocessed_image)
        results = self.model.predict(unprocessed_image, imgsz=640, conf=conf, iou=iou)
        processed_image = None
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    class_name = self.names[int(box.cls)]
                    box = box.xyxy
                    mask = self.sam.predict_sam(unprocessed_image, box)
                    processed_image = draw_image(image_array, mask, box, [class_name])  # Ensure 'draw_image' function is defined
                    image_array = processed_image



        if processed_image is None:
            return unprocessed_image
        else:
            processed_image = Image.fromarray(np.uint8(processed_image)).convert("RGB")
            return processed_image


    def predict_realtime(self):
        cap = cv2.VideoCapture(0) 
        window_name = 'Processed Frame'
        window_created = False  # Track if window exists
        
        # Create the window first
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_created = True
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break  # If the frame is not captured properly, break the loop

            # Check if window was closed with X button (only after window is created)
            if window_created:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Window was destroyed
                    break

            unprocessed_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_array = np.asarray(unprocessed_image)
            processed_image = None
            results = self.model.predict(unprocessed_image, imgsz=640, stream_buffer=True)
            print(len(results[0].boxes))
            
            if len(results[0].boxes) > 0:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_name = self.names[int(box.cls)]
                        box = box.xyxy
                        mask = self.sam.predict_sam(unprocessed_image, box)
                        processed_image = draw_image(image_array, mask, box, [class_name])
                        image_array = processed_image

                processed_image = Image.fromarray(np.uint8(processed_image)).convert("RGB")
                display_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_image)
            else:
                cv2.imshow(window_name, frame) 

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()

    
    def test(self):
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while cap.isOpened():
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Processed Frame', gray)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
        
    def predict_video(self, video, conf=0.3, iou=0.25):
        import os
        import tempfile
        
        print(f"[DEBUG] Starting video prediction for: {video}")
        
        # Use absolute path for output
        output_dir = os.path.abspath(".")  # or use tempfile.gettempdir()
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        
        # Remove old file if exists
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        
        cap = cv2.VideoCapture(video)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps > 0 else 30
        
        print(f"[DEBUG] Video properties: {frame_width}x{frame_height} @ {fps}fps")
        
        # Use mp4v codec which works reliably
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            print("[ERROR] Failed to open video writer")
            return None
        
        # Process video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            
            image_array = np.asarray(image_pil)
            results = self.model.predict(image_pil, imgsz=640, conf=conf, iou=iou)
            processed_image = None
            
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        class_name = self.names[int(box.cls)]
                        box = box.xyxy
                        mask = self.sam.predict_sam(image_pil, box)
                        processed_image = draw_image(image_array, mask, box, [class_name])
                        image_array = processed_image

            if processed_image is None:
                out.write(frame)
            else:
                image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                out.write(image_bgr)
            
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Verify file exists and has content
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            print(f"[SUCCESS] Video saved to: {output_video_path} (size: {os.path.getsize(output_video_path)} bytes)")
            return output_video_path
        else:
            print("[ERROR] Output video file is empty or doesn't exist")
            return None
        

if __name__ == "__main__":
    sam = efficientViT_SAM()
    test = Realtime_Lang_Sam(sam)
    test.init_model(['laptop'])
    test.predict_realtime()
    #test.test()