from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # downloads automatically on first run

# Export to TFLite (int8 quantized for best Pi performance)
model.export(
    format="tflite",
    #imgsz=320        # smaller = faster on Pi; 320 is a good edge trade-off
    #int8=True,        # quantize to INT8 — big speed boost on ARM
    #data="coco8.yaml" # needed for INT8 calibration
)