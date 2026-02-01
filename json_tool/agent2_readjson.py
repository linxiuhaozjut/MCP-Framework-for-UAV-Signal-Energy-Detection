import json


def run_yolo(image_path):
    print(f"Running YOLO on {image_path}")
    return f"YOLO processed {image_path}"

def run_ssd(image_path):
    print(f"Running SSD on {image_path}")
    return f"SSD processed {image_path}"

def run_fasterrcnn(image_path):
    print(f"Running FasterRCNN on {image_path}")
    return f"FasterRCNN processed {image_path}"

def run_maskrcnn(image_path):
    print(f"Running MaskRCNN on {image_path}")
    return f"MaskRCNN processed {image_path}"

def run_dino(image_path):
    print(f"Running DINO on {image_path}")
    return f"DINO processed {image_path}"

def run_dert(image_path):
    print(f"Running DERT on {image_path}")
    return f"DERT processed {image_path}"


def main(json_path="agent2_request.json"):
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data.get("Model", "").lower()
    parameters = data.get("Parameters", {})
    image_path = parameters.get("tf_image", "")

    if not model_name or not image_path:
        print("Model name or image path missing in JSON.")
        return

    model_funcs = {
        "yolo": run_yolo,
        "ssd": run_ssd,
        "fasterrcnn": run_fasterrcnn,
        "maskrcnn": run_maskrcnn,
        "dino": run_dino,
        "dert": run_dert
    }

    func = model_funcs.get(model_name)
    if func:
        result = func(image_path)
        print("Model Output:", result)
    else:
        print(f"Unknown model: {model_name}")


if __name__ == "__main__":
    main("agent2_request.json")
