"""
Categorize conversations using LLM based on extracted keywords (embed-reuse pipeline output)

Inputs:
- s2_keywords_pipeline_embedreuse.json: produced by tools/extract_keywords_conv_embedreuse.py
  Format: [{ conversation_id, conversation_title, response_count, keywords: [{keyword, score}, ...] }, ...]
- Categories: JSON file with an array of category objects or strings. Example:
  [
    {"name": "Bug Report", "description": "User reports a defect or error."},
    {"name": "Feature Request", "description": "User asks for a new capability."},
    "How-To"
  ]

Output:
- JSON with assignments per conversation_id:
  {
    "assignments": {
      "<conversation_id>": {"category": "name", "confidence": 0.87}
    }
  }

Environment:
- Requires OpenAI API key in env var OPENAI_API_KEY (or pass --api-key)
- Default model: gpt-4o-mini (override via --model)
"""
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

try:
    from openai import OpenAI  # openai>=1.0.0 style
except Exception:
    OpenAI = None  # Will check at runtime


@dataclass
class Category:
    name: str
    description: str = ""


def load_categories(path_or_json: str) -> List[Category]:
    """Load categories from a JSON file path or an inline JSON string.

    Tries, in order:
    1) As-is path (relative to CWD)
    2) Path relative to project root (parent of this file)
    3) Parse as inline JSON string
    """
    attempted: List[Path] = []
    # 1) As-is path
    p1 = Path(path_or_json)
    attempted.append(p1)
    if p1.exists():
        data = json.loads(p1.read_text(encoding="utf-8").strip() or "[]")
    else:
        # 2) Relative to project root (parent of this script's dir)
        script_dir = Path(__file__).resolve().parent
        root_rel = (script_dir.parent / path_or_json)
        attempted.append(root_rel)
        if root_rel.exists():
            data = json.loads(root_rel.read_text(encoding="utf-8").strip() or "[]")
        else:
            # 3) Inline JSON
            data = json.loads(path_or_json)
    cats: List[Category] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                cats.append(Category(name=item, description=""))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("category") or ""
                desc = item.get("description") or ""
                if name:
                    cats.append(Category(name=name, description=desc))
    else:
        raise ValueError("Categories must be a JSON array of strings or objects")
    if not cats:
        paths = ", ".join(str(p) for p in attempted)
        raise ValueError(f"No categories provided. Checked paths: {paths}. If passing inline JSON, ensure it is a non-empty JSON array.")
    return cats


def load_keywords(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def conv_brief(item: Dict[str, Any], max_kw: int) -> Dict[str, Any]:
    kws = item.get("keywords", []) or []
    kws_sorted = sorted(kws, key=lambda x: float(x.get("score", 0)), reverse=True)[:max_kw]
    return {
        "conversation_id": item.get("conversation_id"),
        "title": item.get("conversation_title", ""),
        "keywords": [k.get("keyword", "") for k in kws_sorted if k.get("keyword")],
    }


def build_system_prompt(categories: List[Category]) -> str:
    obj = [
        {"name": c.name, "description": c.description} if c.description else c.name
        for c in categories
    ]
    return (
        "You are a highly skilled data scientist making final categorization decisions.\n\n"
        "The categories are:\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n\n"
        "You have preliminary keywords for each conversation. Now you must:\n"
        "1. Review ALL conversations in the current batch holistically\n"
        "2. Consider the overall distribution across categories\n"
        "3. Make final assignment decisions ensuring each conversation is assigned to exactly ONE category\n"
        "4. Adjust confidence scores based on the complete context\n\n"
        "Return JSON only in this format:\n"
        "{\n  \"assignments\": {\n    \"conversation_id\": {\n      \"category\": \"final category\",\n      \"confidence\": 0.90\n    },\n    ...\n  }\n}\n"
    )


def build_user_prompt(batch_items: List[Dict[str, Any]]) -> str:
    # Provide compact JSON to reduce tokens
    compact = [
        {
            "conversation_id": it["conversation_id"],
            "title": it.get("title", ""),
            "keywords": it.get("keywords", []),
        }
        for it in batch_items
    ]
    return json.dumps({"conversations": compact}, ensure_ascii=False, indent=2)


def call_llm(client, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    # OpenAI SDK v1 style
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def parse_assignments(text: str) -> Dict[str, Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        # Try to trim code fences if present
        text2 = text.strip()
        if text2.startswith("```"):
            text2 = text2.strip("`\n")
        obj = json.loads(text2)
    assignments = obj.get("assignments") or {}
    if not isinstance(assignments, dict):
        raise ValueError("LLM response missing 'assignments' object")
    return assignments


def run(
    input_path: Path,
    categories_spec: str,
    output_path: Path,
    model: str,
    api_key: str,
    batch_size: int,
    max_keywords: int,
    temperature: float,
    dry_run: bool,
):
    items = load_keywords(input_path)
    categories = load_categories(categories_spec)

    # Prepare LLM client
    if OpenAI is None:
        raise RuntimeError("openai package not available. Please install openai>=1.0.0")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or use --api-key")
    client = OpenAI(api_key=key)

    # Build system prompt once
    system_prompt = build_system_prompt(categories)

    # Build briefs
    briefs = [conv_brief(it, max_keywords) for it in items]

    # Batch
    all_assignments: Dict[str, Dict[str, Any]] = {}
    for i in tqdm(range(0, len(briefs), batch_size), desc="LLM categorization (batches)"):
        batch = briefs[i:i + batch_size]
        user_prompt = build_user_prompt(batch)
        if dry_run and i == 0:
            # Write the first prompt for inspection
            preview_path = output_path.with_suffix(".prompt_preview.json")
            preview_path.write_text(json.dumps({
                "system": system_prompt,
                "user": user_prompt
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote prompt preview to {preview_path}")
            # Continue to actually call unless dry_run==only
        try:
            raw = call_llm(client, model, system_prompt, user_prompt, temperature)
            assignments = parse_assignments(raw)
        except Exception as e:
            print(f"Batch {i//batch_size}: LLM call/parse failed: {e}")
            continue
        # Merge
        all_assignments.update({str(k): v for k, v in assignments.items()})

    # Write result
    output = {"assignments": all_assignments}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved assignments to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Categorize conversations from s2_keywords_pipeline_embedreuse.json using an LLM")
    p.add_argument("--input", type=str, default="test/output_full/s2_keywords_pipeline_embedreuse.json")
    p.add_argument("--categories", type=str, required=True, help="Path to categories JSON or inline JSON array")
    p.add_argument("--output", type=str, default="test/output_full/s3_llm_categories.json")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-keywords", type=int, default=8, help="max keywords to send per conversation")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true", help="write first batch prompt preview and still run")
    args = p.parse_args()

    run(
        input_path=Path(args.input),
        categories_spec=args.categories,
        output_path=Path(args.output),
        model=args.model,
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_keywords=args.max_keywords,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
