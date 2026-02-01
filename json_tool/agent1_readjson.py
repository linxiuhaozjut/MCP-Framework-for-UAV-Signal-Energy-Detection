import json

from mcp_tools import readbin
from mcp_tools import denoise
from mcp_tools import stfttool




save_dir = "../data/img"
FS = 50e6
CUTOFF_FREQ = 10e6
SLICE_DURATION = 1
OVERLAP_RATIO = 0.1


def rf_denoise(rf_stream):
    iq_data = readbin.load_iq_data(rf_stream)
    filter_taps = denoise.design_lowpass_filter(FS, CUTOFF_FREQ)
    result = denoise.filter_iq_signal(iq_data, filter_taps)
    return result

def stft(rf_stream):
    stft_img = stfttool.iq_to_stft_image(rf_stream, FS)
    stfttool.slice_stft_image_to_squares(stft_img, save_dir)
    print("Done")
    return None


TOOL_REGISTRY = {
    "rf_denoise": rf_denoise,
    "stft": stft
}


json_path = "agent1_request.json"
with open(json_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

tools_sequence = cfg["Tools"][0]              # ["rf_denoise", "stft"]
parameters = cfg["Parameters"]


results = {}


for tool_name in tools_sequence:
    print(f"\nTool: {tool_name} ===")

    tool_func = TOOL_REGISTRY.get(tool_name)
    if tool_func is None:
        raise ValueError(f"NoDefine{tool_name}")


    param_dict = parameters.get(tool_name, {})


    parsed_params = {}

    for key, value in param_dict.items():
        if isinstance(value, str) and value.startswith("results."):
            prev_tool = value.split(".")[1]
            if prev_tool not in results:
                raise ValueError(f"fail_{prev_tool} ")
            parsed_params[key] = results[prev_tool]
        else:
            parsed_params[key] = value


    output = tool_func(**parsed_params)


    results[tool_name] = output



