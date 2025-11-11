"""
Preprocess-only preview for conversation_id=2
- Removes emojis, code blocks, inline code, URLs, markdown noise
- Optionally segments Chinese with jieba
- Shows original/cleaned lengths, cleaned sample, token counts/samples
"""
import json
import re
import sys
from pathlib import Path
from typing import List

try:
    import jieba
    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False

from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH = Path("test/output_full/s1_ai_responses.json")
DEBUG_PRINT_LENGTH = 240


def clean_text(text: str) -> str:
    # Remove code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code `...`
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove emojis (common unicode ranges)
    emoji_pattern = re.compile(
        "["  # start group
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "]",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Markdown/separators/specials cleanup
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"[~#>`]+", " ", text)
    text = text.replace("“", " ").replace("”", " ").replace("’", " ").replace("‘", " ")

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_chinese(text: str) -> str:
    if not JIEBA_AVAILABLE:
        return text
    return " ".join(jieba.cut(text))


def load_conv2_responses(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if x.get("conversation_id") == 2]


def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)

    responses = load_conv2_responses(INPUT_PATH)
    if not responses:
        print("No responses found for conversation_id=2")
        sys.exit(0)

    # Multilingual token pattern:
    # - CJK: >=1 char
    # - Hangul: >=2 chars
    # - Latin/number: >=2 chars
    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"
    vectorizer = CountVectorizer(token_pattern=token_pattern, ngram_range=(1, 3))
    analyzer = vectorizer.build_analyzer()

    for i, resp in enumerate(responses):
        content = resp.get("content", "")
        cleaned = clean_text(content)
        has_chinese_post = any("\u4e00" <= ch <= "\u9fff" for ch in cleaned)

        print("\n=== Response", i, "===")
        print("original_len:", len(content), "cleaned_len:", len(cleaned), "has_chinese_post:", has_chinese_post)
        print("cleaned_sample:", cleaned[:DEBUG_PRINT_LENGTH])

        tokens = analyzer(cleaned)
        print("tokens_count:", len(tokens), "tokens_sample:", tokens[:50])

        # If tokens empty, fallback: segment original content
        if len(tokens) == 0:
            if JIEBA_AVAILABLE:
                rebuilt = " ".join(jieba.cut(content))
            else:
                def _char_map(ch: str) -> str:
                    if ('\u4e00' <= ch <= '\u9fff') or ('\u3131' <= ch <= '\u318e') or ('\uac00' <= ch <= '\ud7a3'):
                        return ch + ' '
                    if ch.isalnum() or ch == '_':
                        return ch
                    return ' '
                rebuilt = ''.join(_char_map(ch) for ch in content)
                rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
            print("[fallback] rebuilt_sample:", rebuilt[:DEBUG_PRINT_LENGTH])
            tokens2 = analyzer(rebuilt)
            print("[fallback] tokens_count:", len(tokens2), "tokens_sample:", tokens2[:50])

if __name__ == "__main__":
    main()
