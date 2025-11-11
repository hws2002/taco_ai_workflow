"""
Build s1_ai_responses.json from data/conversations.json
- Samples N conversations (default: 50) via --sample
- Produces test/output_full/s1_ai_responses.json by default
- Response schema per item:
  {
    "response_id": "<conversation_id>_<message_index>",
    "conversation_id": <int|str>,
    "conversation_title": <str>,
    "message_index": <int>,
    "content": <str>,
    "timestamp": <float|int|str|null>
  }
Supports flexible input schema:
- conversation id: conversation_id | id
- title: conversation_title | title | name
- responses/messages list keys: responses | messages | items
- message content: content | text | body
- message timestamp: timestamp | time | created_at
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _pick(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_int(s):
    try:
        return int(s)
    except Exception:
        return s


def load_conversations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either list or {"conversations": [...]}
    if isinstance(data, dict) and "conversations" in data:
        data = data["conversations"]
    if not isinstance(data, list):
        raise ValueError("Input must be a list of conversations or an object with 'conversations' list")
    return data


def iter_messages(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("responses", "messages", "items"):
        if key in conv and isinstance(conv[key], list):
            return conv[key]
    return []


def build_s1(
    input_path: Path,
    output_path: Path,
    sample: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    conversations = load_conversations(input_path)
    rng = random.Random(seed)

    # sample conversation indices
    if sample and sample < len(conversations):
        indices = rng.sample(range(len(conversations)), sample)
        selected = [conversations[i] for i in sorted(indices)]
    else:
        selected = conversations

    out: List[Dict[str, Any]] = []

    for conv in selected:
        conv_id = _pick(conv, ["conversation_id", "id"])
        conv_id = _as_int(conv_id)
        title = _pick(conv, ["conversation_title", "title", "name"], default="")
        msgs = iter_messages(conv)

        for idx, msg in enumerate(msgs):
            content = _pick(msg, ["content", "text", "body"], default="")
            ts = _pick(msg, ["timestamp", "time", "created_at"], default=None)
            out.append({
                "response_id": f"{conv_id}_{idx}",
                "conversation_id": conv_id,
                "conversation_title": title,
                "message_index": idx,
                "content": content,
                "timestamp": ts,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    p = argparse.ArgumentParser(description="Build s1_ai_responses.json from conversations.json (with sampling)")
    p.add_argument("--input", type=str, default="data/conversations.json", help="input conversations.json path")
    p.add_argument("--output", type=str, default="test/output_full/s1_ai_responses.json", help="output s1 json path")
    p.add_argument("--sample", type=int, default=50, help="number of conversations to sample (default: 50)")
    p.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    args = p.parse_args()

    build_s1(Path(args.input), Path(args.output), sample=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
