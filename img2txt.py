from PIL import Image
import os

BLIP_AVAILABLE = False
processor = None
model = None

def img2txt(image):
    global processor, model, BLIP_AVAILABLE
    if processor is None or model is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("Loading Salesforce BLIP model lazily...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
            BLIP_AVAILABLE = True
            print("Salesforce BLIP model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load BLIP captioning model: {e}. Running in Demo/Simulation Mode.")

    if not BLIP_AVAILABLE or model is None or processor is None:
        # Mock description of typical medical or normal chest scans
        w, h = image.size
        val = (w + h) % 3
        if val == 0:
            return "a high resolution frontal chest x-ray showing normal lung fields and cardiac silhouette"
        elif val == 1:
            return "a chest radiograph displaying patch consolidation in the right middle and lower lobes consistent with bacterial pneumonia"
        else:
            return "a frontal chest x-ray showing bilateral diffuse interstitial infiltrates suggestive of viral infection"

    import torch
    raw_image = image.convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs)
    output_string = (processor.decode(out[0], skip_special_tokens=True))
    return output_string


# print(img2txt('uploads/xraytest.jpeg'))