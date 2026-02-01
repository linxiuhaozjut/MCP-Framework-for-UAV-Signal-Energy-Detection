import json
import re

def extract_field(text, field):

    if field == "Parameters":

        pattern = rf'"Parameters"\s*:\s*(\{{.*?\}})'
    else:

        pattern = rf'"{field}"\s*:\s*"([^"]*?)"'

    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None

def parse_json_from_text(text):

    result = {}

    # --- Summary ---
    summary_val = extract_field(text, "Summary")
    if summary_val:
        result["Summary"] = summary_val

    # --- Model ---
    model_val = extract_field(text, "Model")
    if model_val:
        result["Model"] = model_val.lower()

    # --- Parameters ---
    parameters_val = extract_field(text, "Parameters")
    if parameters_val:
        try:

            parameters_dict = json.loads(parameters_val.replace("\\", ""))
        except:
            parameters_dict = {}
        result["Parameters"] = parameters_dict

    return result

def save_json_from_text(text, output_path="agent2_request.json"):
    data = parse_json_from_text(text)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON to {output_path}")


if __name__ == "__main__":

    with open("input.txt", "r", encoding="utf-8") as f:
        content = f.read()

    save_json_from_text(content)
