import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse

from dotenv import load_dotenv

# OpenAI client (use the official openai package)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def load_keywords(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_keywords_per_conversation(items: List[Dict[str, Any]], top_n_per_response: int = 5) -> tuple[Dict[Any, List[str]], Dict[Any, str]]:
    conv_kw: Dict[Any, List[str]] = {}
    conv_titles: Dict[Any, str] = {}

    for it in items:
        cid = it.get('conversation_id')
        kws = it.get('keywords', [])

        # Store conversation title (first occurrence wins)
        if cid not in conv_titles and 'conversation_title' in it:
            conv_titles[cid] = it['conversation_title']

        # take top-N per response to avoid overwhelming the prompt
        picked = [k.get('keyword') for k in kws[:top_n_per_response] if isinstance(k, dict) and 'keyword' in k]
        if not picked:
            continue
        conv_kw.setdefault(cid, []).extend(picked)

    # de-duplicate and keep order per conversation
    for cid, arr in conv_kw.items():
        seen = set()
        deduped = []
        for k in arr:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        conv_kw[cid] = deduped

    return conv_kw, conv_titles


def load_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def call_llm_categories(system_prompt: str, user_prompt: str, model: str, provider: str = 'openai') -> str:
    if provider != 'openai':
        raise ValueError(f"Unsupported provider: {provider}")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install with: pip install openai python-dotenv")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Create a .env with OPENAI_API_KEY=... or export it.")

    client = OpenAI(
        api_key=api_key
        # base_url="https://api.chatanywhere.tech/v1"
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def main():
    project_root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="LLM-based coarse categorization using extracted keywords")
    parser.add_argument("--keywords-input", type=str, default=str(project_root / 'test' / 'output' / 's2_ai_responses_with_keywords.json'), help="키워드 JSON 경로")
    parser.add_argument("--prompt-file", type=str, default=str(project_root / 'llm_clustering_prompt.txt'), help="프롬프트 템플릿 파일 경로")
    parser.add_argument("--output", type=str, default=str(project_root / 'test' / 'output' / 's6_categories.json'), help="대분류 결과 JSON 출력 경로")
    parser.add_argument("--assignments-output", type=str, default=str(project_root / 'test' / 'output' / 's6_categories_assignments.json'), help="assignments 전용 매핑 JSON 출력 경로 (topic_pipeline --categories 용)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM 모델명 (예: gpt-4o)")
    parser.add_argument("--provider", type=str, default="openai", help="LLM 제공자 (현재 openai만 지원)")
    parser.add_argument("--top-n-per-response", type=int, default=5, help="응답당 사용할 상위 키워드 수")
    parser.add_argument("--raw-output", type=str, default=str(project_root / 'test' / 'output' / 's6_categories_raw.txt'), help="LLM 원문 응답 저장 경로")
    args = parser.parse_args()

    # .env 로드
    load_dotenv()

    # 데이터 로드 및 집계
    kw_items = load_keywords(args.keywords_input)
    conv_keywords, conv_titles = aggregate_keywords_per_conversation(kw_items, top_n_per_response=args.top_n_per_response)

    # 프롬프트 구성
    system_prompt = load_prompt(args.prompt_file)
    # Provide explicit JSON output guidance (include reason)
    # output_schema_hint = (
    #     "\n\n요청: 다음 형식의 JSON만 반환하세요. 예시:\n" \
    #     "{\n  \"categories\": [\"카테고리1\", \"카테고리2\", ...],\n  \"assignments\": { \"<conversation_id>\": \"카테고리명\", ... },\n  \"reason\": \"왜 이런 카테고리와 할당을 했는지에 대한 상세 설명\"\n}\n"
    # )

    # Prepare a compact JSON of conversation->keywords for the model
    preview = {
        str(cid): kws[:50]  # safety cap per conversation
        for cid, kws in conv_keywords.items()
    }
    user_payload = json.dumps(preview, ensure_ascii=False, indent=2)

    user_prompt = f"[입력 데이터: 대화별 키워드]\n{user_payload}"

    # LLM 호출
    print("LLM 대분류 실행 중...")
    content = call_llm_categories(system_prompt, user_prompt, model=args.model, provider=args.provider)

    # 응답 파싱
    try:
        data = json.loads(content)
    except Exception as e:
        # 응답이 코드블록 등에 감싸진 경우 대비 간단 정리
        cleaned = content.strip().strip('`')
        try:
            data = json.loads(cleaned)
        except Exception:
            raise RuntimeError(f"LLM 응답을 JSON으로 파싱하지 못했습니다. 원문:\n{content}")

    # 저장 (JSON 결과)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # assignments-only 저장 (conversation_title 포함)
    assignments = data.get('assignments', {}) if isinstance(data, dict) else {}
    # Add conversation_title to each assignment
    enriched_assignments = {}
    for conv_id, assignment in assignments.items():
        enriched_assignment = assignment.copy() if isinstance(assignment, dict) else {"category": assignment, "confidence": 1.0}
        enriched_assignment['conversation_title'] = conv_titles.get(int(conv_id) if conv_id.isdigit() else conv_id, f"Conversation {conv_id}")
        enriched_assignments[conv_id] = enriched_assignment

    assign_path = Path(args.assignments_output)
    assign_path.parent.mkdir(parents=True, exist_ok=True)
    with open(assign_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_assignments, f, ensure_ascii=False, indent=2)

    # LLM 원문 응답 별도 저장
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"대분류 결과 저장: {out_path}")
    print(f"카테고리 매핑 저장(--categories 용): {assign_path}")
    print(f"LLM 원문 저장: {raw_path}")


if __name__ == "__main__":
    main()
