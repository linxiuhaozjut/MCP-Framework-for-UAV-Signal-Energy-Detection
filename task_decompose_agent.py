import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, CLIPProcessor, CLIPModel
from PIL import Image
from rag_moduel.rag_retrieved import multimodal_rag_retrieve
from json_tool.agent2_jsonsave import save_json_from_text

# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
llava_model_dir = "llava-1.5-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(llava_model_dir)
llava_model = LlamaForCausalLM.from_pretrained(
    llava_model_dir,
    torch_dtype=torch.float16,
    device_map={"": device}
)

# ==========================================
clip_model_dir = "clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_dir).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)

# ==========================================
image_path = "..."
image = Image.open(image_path).convert("RGB")

# ==========================================
text_prompt = "Which models is the suitable?"

# ==========================================
retrieved_images, retrieved_texts = multimodal_rag_retrieve(image, text_prompt, topk=5)


rag_context = ""
for img_path, txt in zip(retrieved_images, retrieved_texts):
    rag_context += f"<image>{img_path}</image>\n{txt}\n"

# ==========================================
prompt = f"""
{rag_context}\nQuery: {text_prompt}
Answer Query as Summary.
Choose a suitable model from the ModelList.ModelList：["yolo","ssd","fasterrcnn","maskrcnn","dino","dert"]
Provide the configuration parameters for selected model.
{{"tf_image":"{image_path}"}}

Replace the above parameters into the JSON below.
The output will only be in the following format:
{{
  "Summary": "...",
  "Model": "yolo",
  "Parameters": {{"tf_image":"image.jpg"}}
}}
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ==========================================
with torch.no_grad():
    output_ids = llava_model.generate(
        input_ids=inputs["input_ids"],
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("===== LLaVA Output =====")
print(output_text)

save_json_from_text(output_text)
