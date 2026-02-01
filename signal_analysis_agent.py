import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
import json
from jsontool.agent1_jsonsave import save_json_from_text

# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================LLaVA=====================
llava_model_dir = "./llava-1.5-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(llava_model_dir)

llava_model = LlamaForCausalLM.from_pretrained(
    llava_model_dir,
    torch_dtype=torch.float16,
    device_map={"": device}
)

# =====================bin=====================
def read_bin_iq(bin_path):
    raw = np.fromfile(bin_path, dtype=np.int16)
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    I = raw[0::2]
    Q = raw[1::2]
    return I.astype(np.float32) + 1j * Q.astype(np.float32)

# ==========================================
bin_path = "..."
iq_data = read_bin_iq(bin_path)

# ===================== =====================
iq_text = ", ".join([f"{v.real:.2f}{v.imag:+.2f}j" for v in iq_data[:20]])

json_prompt = f"""
The content of the summary describes whether this sequence needs denoising and then uses STFT.
IQ data snippet is {iq_text}, it should not be written in the summary.
The content of Tools consists of the options in the ToolList below.
ToolList：["rf_denoise","stft"]
Provide the configuration parameters for each selected tool.
{{"rf_denoise": {{"rf_stream": "iq_data"}}, "stft": {{"rf_stream": "iq_data"}}}}

Replace the above parameters into the JSON below.
The output will only be in the following format:
{{
"Summary": "...",
"Tools": ["rf_denoise","stft"],
"Parameters": "rf_denoise": {{"rf_stream":[iq_data]}},"stft": {{"rf_stream":"results.rf_denoise"}}
}}
"""

# =========================================
inputs = tokenizer(json_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = llava_model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)


try:
    json_output = json.loads(output_text)
except json.JSONDecodeError:
    json_output = output_text

print("\n===== LLaVA JSON Output =====")
print(json_output)

save_json_from_text(json_output, bin_path)

