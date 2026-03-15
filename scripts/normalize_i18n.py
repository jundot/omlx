import json
from pathlib import Path

def normalize():
    project_root = Path(__file__).parent.parent
    i18n_dir = project_root / "omlx" / "admin" / "i18n"
    
    en_path = i18n_dir / "en.json"
    if not en_path.exists():
        print(f"Error: Could not find {en_path}")
        return

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    for file_path in i18n_dir.glob("*.json"):
        if file_path.name == "en.json":
            continue
            
        print(f"Normalizing {file_path.name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lang_data = json.load(f)
            
        normalized_data = {}
        for key, value in en_data.items():
            if key in lang_data:
                normalized_data[key] = lang_data[key]
            else:
                normalized_data[key] = value
                
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized_data, f, indent=4, ensure_ascii=False)
            f.write("\n") # Add trailing newline

if __name__ == "__main__":
    normalize()
    print("Done.")
