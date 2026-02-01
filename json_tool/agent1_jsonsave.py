import json
import re


def extract_last_field(text, field):

    pattern = rf'"{field}"\s*:\s*(.*)'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def clean_value(value):

    return value.replace("\\", "")


def parse_json_block(text):
    summary_raw = extract_last_field(text, "Summary")
    tools_raw = extract_last_field(text, "Tools")
    parameters_raw = extract_last_field(text, "Parameters")

    result = {}

    # --- Summary ---
    if summary_raw:
        summary_clean = clean_value(summary_raw.strip().strip('",'))
        result["Summary"] = summary_clean.strip('"')

    # --- Tools ---
    if tools_raw:
        tools_clean = clean_value(tools_raw)
        try:
            tools_list = eval(tools_clean)  #  ["xx","xx"]
        except:
            tools_list = []
        result["Tools"] = tools_list

    # --- Parameters ---
    if parameters_raw:
        parameters_clean = clean_value(parameters_raw)


        if not parameters_clean.strip().startswith("{"):
            parameters_clean = "{" + parameters_clean + "}"

        try:
            parameters_dict = eval(parameters_clean)
        except:
            parameters_dict = {}

        result["Parameters"] = parameters_dict

    return result


def replace_iq_data(parameters_dict, bin_path):

    if isinstance(parameters_dict, dict):
        for k, v in parameters_dict.items():
            if isinstance(v, dict):
                replace_iq_data(v, bin_path)
            elif v == "iq_data":
                parameters_dict[k] = bin_path
    elif isinstance(parameters_dict, list):
        for i, item in enumerate(parameters_dict):
            if item == "iq_data":
                parameters_dict[i] = bin_path
            elif isinstance(item, dict) or isinstance(item, list):
                replace_iq_data(item, bin_path)


def save_json_from_text(text, bin_path, output_path="agent1_request.json"):
    data = parse_json_block(text)


    if "Parameters" in data:
        replace_iq_data(data["Parameters"], bin_path)


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON to {output_path}")



if __name__ == "__main__":
    bin_path_value = "/path/to/your/bin/file"
    with open("input.txt", "r", encoding="utf-8") as f:
        content = f.read()
    save_json_from_text(content, bin_path_value)
