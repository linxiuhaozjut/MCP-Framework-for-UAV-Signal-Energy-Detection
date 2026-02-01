import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel
from PIL import Image
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===================== 加载知识库 =====================
with open("multimodal_knowledge.pkl", "rb") as f:
    knowledge = pickle.load(f)

knowledge_vecs = knowledge["vectors"].to(device)  # [N, dim]
knowledge_data = knowledge["data"]  # [{"image": path, "text": txt}, ...]

# ===================== 加载 CLIP =====================
clip_model_dir = "/data/Newdisk/Bigmodel/zxm/mcp-vision/model/clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)
vision_model = CLIPVisionModel.from_pretrained(clip_model_dir).to(device)
text_model = CLIPTextModel.from_pretrained(clip_model_dir).to(device)

# ===================== 定义线性投影 =====================
vision_to_text_proj = nn.Linear(1024, 768).to(device)
vision_to_text_proj.eval()  # 推理阶段不训练

# ===================== 检索函数封装 =====================
def multimodal_rag_retrieve(query_image: Image.Image, query_text: str, topk: int = 5):
    """
    输入：
        query_image: PIL.Image 查询图像
        query_text: str 查询文本
        topk: int 检索返回数量
    输出：
        retrieved_image_paths: list[str] 检索到的图像路径列表
        retrieved_texts: list[str] 检索到的文本列表
    """
    # -------- CLIP 特征提取 --------
    # 图像
    vision_inputs = clip_processor(images=query_image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = vision_model(**vision_inputs)
        image_feat = vision_outputs.pooler_output  # [1, 1024]

    # 文本
    text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = text_model(**text_inputs)
        text_feat = text_outputs.pooler_output  # [1, 768]

    # -------- 线性投影 + 融合 --------
    image_feat_proj = vision_to_text_proj(image_feat)  # [1, 768]
    image_feat_proj = F.normalize(image_feat_proj, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    query_vec = F.normalize(0.5 * image_feat_proj + 0.5 * text_feat, dim=-1)  # 融合向量

    # -------- 检索 top-k --------
    cos_sim = torch.matmul(query_vec, knowledge_vecs.T)  # [1, N]
    values, indices = torch.topk(cos_sim, topk, dim=-1)

    retrieved_data = [knowledge_data[i] for i in indices[0]]
    retrieved_texts = [d["text"] for d in retrieved_data]
    retrieved_image_paths = [d["image"] for d in retrieved_data]

    return retrieved_image_paths, retrieved_texts

# ===================== 测试函数 =====================
if __name__ == "__main__":
    query_image_path = "/data/Newdisk/Bigmodel/zxm/mcp-vision/mcpsignal/data/img/stft_img_0.jpg"
    query_text = "Which model is the best?."
    query_image = Image.open(query_image_path).convert("RGB")

    retrieved_images, retrieved_texts = multimodal_rag_retrieve(query_image, query_text, topk=5)

    print("===== Retrieved Knowledge =====")
    for img_path, txt in zip(retrieved_images, retrieved_texts):
        print(f"Image: {img_path} | Text: {txt}")
