import json
import re
from pathlib import Path

def normalize():
    project_root = Path(__file__).parent.parent
    i18n_dir = project_root / "omlx" / "admin" / "i18n"
    
    en_path = i18n_dir / "en.json"
    if not en_path.exists():
        print(f"Error: Could not find {en_path}")
        return

    with open(en_path, "r", encoding="utf-8") as f:
        en_text = f.read()
        
    en_data = json.loads(en_text)
    
    line_pattern = re.compile(r'^(\s*)"([^"]+)"(\s*:\s*)(.*)$')

    for file_path in i18n_dir.glob("*.json"):
        if file_path.name == "en.json":
            continue
            
        print(f"Normalizing {file_path.name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lang_data = json.load(f)
            
        normalized_lines = []
        for line in en_text.splitlines():
            match = line_pattern.match(line)
            if match:
                indent, key, colon_space, old_val_part = match.groups()
                target_value = lang_data.get(key, en_data[key])
                json_value = json.dumps(target_value, ensure_ascii=False)
                
                has_comma = old_val_part.strip().endswith(',')
                if has_comma:
                    json_value += ","
                    
                new_line = f'{indent}"{key}"{colon_space}{json_value}'
                normalized_lines.append(new_line)
            else:
                normalized_lines.append(line)
                
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized_lines) + "\n")

if __name__ == "__main__":
    normalize()
    print("Done.")
