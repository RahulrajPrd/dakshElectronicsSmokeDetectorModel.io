import os
import cv2
import fitz
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def convert_pdf_to_image(pdf_path, output_image_path, dpi=700):
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(output_image_path)
        doc.close()
        return True
    except:
        return False

def initialize_yolo_model(model_path, confidence_threshold = 0.3):
    device = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    return AutoDetectionModel.from_pretrained(
        model_type="yolov11",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )

# customize the slice width and height as per the requirement of the image file, small size -> better results but slow, large size -> good results and faster
def perform_sliced_prediction(image_path, detection_model, slice_height=1500, slice_width = 1500, overlap_height_ratio = 0.2, overlap_width_ratio = 0.2):
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )

    return result

def extract_coordinates(prediction_result):
    detections = []
    for obj in prediction_result.object_prediction_list:
        detection_type = obj.category.name
        bbox = obj.bbox.box
        center_x = (float(bbox[0]) + float(bbox[2])) / 2
        center_y = (float(bbox[1]) + float(bbox[3])) / 2
        detections.append([detection_type, [center_x, center_y]])
    return detections

def process_pdf_with_yolo(pdf_path, model_path, output_image_path):
    if not convert_pdf_to_image(pdf_path, output_image_path):
        raise RuntimeError("Failed to convert PDF to image")
    
    detection_model = initialize_yolo_model(model_path)
    prediction_result = perform_sliced_prediction(output_image_path, detection_model)

    return extract_coordinates(prediction_result)

if __name__ == "__main__":
    pdf_path = "PDFs/basement.pdf" # example PDF is present in the PDFs folder
    model_path = "model/best.pt"  # The actual model path
    output_image_path = "images.png"
    
    try:
        results = process_pdf_with_yolo(pdf_path, model_path, output_image_path)
        print(results)
    except Exception as e:
        print(f"Failed to process PDF: {str(e)}")
