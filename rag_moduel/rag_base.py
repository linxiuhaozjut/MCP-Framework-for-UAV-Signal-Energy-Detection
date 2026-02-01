import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel
from PIL import Image
import os
import pickle
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===================== CLIP  =====================
clip_model_dir = "../clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)

#
vision_model = CLIPVisionModel.from_pretrained(clip_model_dir).to(device)
text_model = CLIPTextModel.from_pretrained(clip_model_dir).to(device)

# ==========================================
vision_to_text_proj = nn.Linear(1024, 768).to(device)

# ===================== 数据路径 =====================
images_folder = "./image_data/"
text_file_path = "./text_data/text.txt"

if not os.path.exists(text_file_path):
    raise FileNotFoundError(f"{text_file_path} not found!")

with open(text_file_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

image_files = [os.path.join(images_folder, f"img_{i:02d}.jpg") for i in range(len(texts))]

# ===================== embedding =====================
knowledge_vecs = []
knowledge_data = []

for img_path, txt in zip(image_files, texts):
    #
    image = Image.open(img_path).convert("RGB")

    #
    vision_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = vision_model(**vision_inputs)
        image_feat = vision_outputs.pooler_output  # [1, 1024]

    #
    text_inputs = clip_processor(text=[txt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = text_model(**text_inputs)
        text_feat = text_outputs.pooler_output  # [1, 768]

    # ==========================================
    image_feat_proj = vision_to_text_proj(image_feat)  # [1, 768]

    #
    image_feat_proj = F.normalize(image_feat_proj, p=2, dim=-1)
    text_feat = F.normalize(text_feat, p=2, dim=-1)

    #
    fused_vec = F.normalize(0.8 * image_feat_proj + 0.2 * text_feat, dim=-1)

    #
    knowledge_vecs.append(fused_vec.cpu())
    knowledge_data.append({"image": img_path, "text": txt})

#
knowledge_vecs = torch.cat(knowledge_vecs, dim=0)

#
with open("multimodal_knowledge.pkl", "wb") as f:
    pickle.dump({"vectors": knowledge_vecs, "data": knowledge_data}, f)

print("Knowledge library built and saved!")
