"""
Step 2: Preprocess responses using preprocess.preprocess_content
- Input: test/output_full/s1_ai_responses.json (or --input)
- Output: test/output_full/s1_ai_responses_preprocessed.json (or --output)
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# import user's preprocess function
try:
    from preprocess import preprocess_content
except Exception as e:
    raise RuntimeError(f"Failed to import preprocess.preprocess_content: {e}")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(input_path: Path, output_path: Path):
    rows = load_json(input_path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        content = r.get("content", "")
        cleaned = preprocess_content(content)
        rr = dict(r)
        rr["content"] = cleaned
        out.append(rr)
    save_json(output_path, out)


def main():
    p = argparse.ArgumentParser(description="Preprocess s1 responses with preprocess.preprocess_content")
    p.add_argument("--input", type=str, default="test/output_full/s1_ai_responses.json")
    p.add_argument("--output", type=str, default="test/output_full/s1_ai_responses_preprocessed.json")
    args = p.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
