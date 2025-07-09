import torch
from ultralytics import YOLO
import gc

torch.set_num_threads(4)
torch.backends.quantized.engine = 'qnnpack'

model = YOLO('best.pt')

for _ in model.predict(
        source='car.mp4',
        imgsz=640,
        conf=0.7,
        device='cpu',
        half=False,
        verbose=True,
        stream=True  # critical to avoid storing all results
):
    # Don't even access the result → completely discard it
    gc.collect()  # optional: force garbage collection