"""
통합 키워드 추출 파이프라인
1. AI 응답 추출
2. Conversation별 임베딩 벡터 생성 (mean pooling)
3. Conversation별 키워드 추출 (다국어 지원)
"""

import json
import pickle
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# 필수 라이브러리
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

# 선택적 라이브러리
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not installed. Chinese tokenization will be suboptimal.")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Language detection disabled.")


# ============================================================================
# 유틸리티 함수
# ============================================================================

def detect_language(text: str) -> str:
    """텍스트의 언어를 감지"""
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
        return 'unknown'
    
    try:
        lang = detect(text)
        return lang
    except (LangDetectException, Exception):
        return 'unknown'


def clean_text(text: str) -> str:
    """텍스트 전처리: 코드, 이모티콘, URL 제거"""
    # 코드 블록 제거
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    
    # 이모티콘 제거
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)
    
    # URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 마크다운 구분선/불필요한 기호 제거
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\*+", " ", text)  # ****, ** 등 제거
    text = re.sub(r"[~#>`]+", " ", text)  # 틸드, 헤더, 인용, 백틱 제거
    # 전각 따옴표 등을 공백으로 치환
    text = text.replace("“", " ").replace("”", " ").replace("’", " ").replace("‘", " ")

    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_chinese(text: str) -> str:
    """중국어 텍스트를 jieba로 토크나이징"""
    if not JIEBA_AVAILABLE:
        return text
    
    words = jieba.cut(text)
    return ' '.join(words)


# ============================================================================
# 1. AI 응답 로딩
# ============================================================================

def load_responses(filepath: str) -> List[Dict[str, Any]]:
    """AI 응답 JSON 파일 로드"""
    print(f"AI 응답 로딩 중: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    print(f"총 {len(responses)}개 응답 로드됨\n")
    return responses


def group_by_conversation(responses: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """응답을 conversation_id별로 그룹화"""
    grouped = defaultdict(list)
    for resp in responses:
        conv_id = resp.get('conversation_id')
        if conv_id is not None:
            grouped[conv_id].append(resp)
    return dict(grouped)


# ============================================================================
# 2. 임베딩 벡터 생성 (Conversation 단위)
# ============================================================================

def create_conversation_embeddings(
    grouped_responses: Dict[int, List[Dict[str, Any]]],
    model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
    batch_size: int = 32,
    cache_dir: str = None
) -> Dict[int, np.ndarray]:
    """
    Deprecated: 더 이상 mean pooling 임베딩을 사용하지 않습니다.
    하위 호환을 위해 빈 딕셔너리를 반환합니다.
    """
    print("경고: mean pooling 임베딩은 사용하지 않음 -> 건너뜀")
    return {}


# ============================================================================
# 3. 키워드 추출 (Conversation 단위)
# ============================================================================

def extract_keywords_for_conversation(
    conv_id: int,
    responses: List[Dict[str, Any]],
    conv_embedding: np.ndarray,
    kw_model: KeyBERT,
    top_n: int = 5,
    keywords_per_response: int = 5,
    debug_conv_id: Optional[int] = None,
    debug_print_length: int = 200,
    ensure_cjk: bool = False
) -> tuple:
    """
    하나의 conversation에서 키워드를 추출
    
    방식:
    1. 각 응답마다 키워드 N개씩 추출
    2. 모든 키워드를 모아서 점수 기준으로 정렬
    3. 상위 top_n개만 선택
    
    Returns:
        (conv_id, keywords)
    """
    all_keywords = []
    failed_responses = 0
    processed_responses = 0
    
    # 멀티링구얼 토큰 패턴 (CJK/Hangul/영문/숫자)
    # - CJK(漢字): 1자 이상 허용 (jieba 분절 후 다자 단어도 커버)
    # - Hangul: 2자 이상 (의미성 확보)
    # - 영문/숫자: 2자 이상
    # - 하이픈 전용 토큰은 배제 ('-' 미포함)
    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"
    vectorizer = CountVectorizer(token_pattern=token_pattern, ngram_range=(1, 3))

    # 각 응답마다 키워드 추출
    for idx, resp in enumerate(responses):
        content = resp.get('content', '')
        if not content or len(content.strip()) < 20:
            continue
        
        # 텍스트 전처리
        cleaned_text = clean_text(content)
        
        if len(cleaned_text.strip()) < 20:
            continue
        
        processed_responses += 1
        
        # 중국어 토크나이징 (중국어가 포함되어 있으면 처리)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in cleaned_text)
        if has_chinese and JIEBA_AVAILABLE:
            cleaned_text = tokenize_chinese(cleaned_text)

        # 디버그 출력: 전처리 결과와 토큰/후보 확인
        if debug_conv_id is not None and conv_id == debug_conv_id:
            print("\n--- DEBUG: Conversation", conv_id, "Response", idx, "---")
            print("original_len:", len(content), "cleaned_len:", len(cleaned_text), "has_chinese:", has_chinese)
            print("cleaned_text_sample:", cleaned_text[:debug_print_length].replace("\n", " "), "...")
            analyzer = vectorizer.build_analyzer()
            tokens = analyzer(cleaned_text)
            print("tokens_count:", len(tokens), "tokens_sample:", tokens[:50])
            # ngram 후보 개수/상위 TF 일부 출력
            try:
                dbg_vec = CountVectorizer(token_pattern=vectorizer.token_pattern, ngram_range=(1, 3))
                X = dbg_vec.fit_transform([cleaned_text])
                feats = dbg_vec.get_feature_names_out()
                counts = np.asarray(X.sum(axis=0)).ravel()
                order = counts.argsort()[::-1]
                top_idx = order[:10]
                top_feats = [(feats[i], int(counts[i])) for i in top_idx]
                print("candidates:", len(feats), "top_tf:", top_feats)
            except Exception as e:
                print("candidate_debug_error:", str(e))
        
        # Fallback: 토큰이 비면 원본문 기반 CJK 보강 (언어 감지 무관, 무조건 보강 시도)
        analyzer = vectorizer.build_analyzer()
        tokens_probe = analyzer(cleaned_text)
        if len(tokens_probe) == 0:
            # 1) jieba 가능하면 원문에 대해 세그먼트
            if JIEBA_AVAILABLE:
                rebuilt = ' '.join(jieba.cut(content))
            else:
                # 2) CJK 문자에 공백 삽입, 한글/영문/숫자 토큰은 그대로 보존
                def _char_map(ch: str) -> str:
                    if ('\u4e00' <= ch <= '\u9fff') or ('\u3131' <= ch <= '\u318e') or ('\uac00' <= ch <= '\ud7a3'):
                        return ch + ' '
                    if ch.isalnum() or ch == '_':
                        return ch
                    return ' '
                rebuilt = ''.join(_char_map(ch) for ch in content)
                rebuilt = re.sub(r'\s+', ' ', rebuilt).strip()
            cleaned_text = rebuilt
            if debug_conv_id is not None and conv_id == debug_conv_id:
                print("[fallback] rebuilt_for_cjk sample:", cleaned_text[:debug_print_length])
            # fallback 후 토큰 재계산 디버그
            if debug_conv_id is not None and conv_id == debug_conv_id:
                tokens_after = analyzer(cleaned_text)
                print("[fallback] tokens_count:", len(tokens_after), "tokens_sample:", tokens_after[:50])
        
        try:
            # KeyBERT 실행 - 각 응답마다
            extracted = kw_model.extract_keywords(
                cleaned_text,
                keyphrase_ngram_range=(1, 3),
                stop_words=None,
                use_mmr=True,
                diversity=0.5,
                nr_candidates=30,
                top_n=keywords_per_response,
                vectorizer=vectorizer
            )
            
            # 키워드 수집
            for kw, score in extracted:
                all_keywords.append({
                    'keyword': kw,
                    'score': float(score),
                    'response_index': idx
                })
                    
        except Exception as e:
            failed_responses += 1
            continue
    
    # 키워드가 하나도 없으면 실패
    if len(all_keywords) == 0:
        print(f"  ❌ Conv {conv_id}: 키워드 추출 실패")
        print(f"     └─ 총 응답: {len(responses)}개, 처리된 응답: {processed_responses}개")
        print(f"     └─ 실패한 응답: {failed_responses}개")
        if processed_responses > 0:
            print(f"     └─ 원인: 처리된 응답에서 키워드가 추출되지 않음 (텍스트가 너무 짧거나 특수문자만 있을 가능성)")
        else:
            print(f"     └─ 원인: 처리 가능한 응답 없음 (모든 응답이 20자 미만)")
        return (conv_id, [])
    
    # 중복 키워드 제거하면서 최고 점수만 유지
    keyword_scores = {}
    for kw_info in all_keywords:
        kw = kw_info['keyword']
        score = kw_info['score']
        
        if kw not in keyword_scores or score > keyword_scores[kw]['score']:
            keyword_scores[kw] = kw_info
    
    # CJK/Hangul 선호도 반영 및 선택
    def has_cjk(s: str) -> bool:
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    def has_hangul(s: str) -> bool:
        return any('\u3131' <= ch <= '\u318e' or '\uac00' <= ch <= '\ud7a3' for ch in s)

    candidates = list(keyword_scores.values())

    # 선택적 가중치: CJK/Hangul 후보에 소폭 보너스
    biased = []
    for item in candidates:
        bonus = 0.0
        if has_cjk(item['keyword']) or has_hangul(item['keyword']):
            bonus += 0.05
        biased.append((item['score'] + bonus, item))

    biased.sort(key=lambda t: t[0], reverse=True)
    picked = [it[1] for it in biased]

    # 상위 top_n 우선 선택
    final = picked[:top_n]

    # 옵션: CJK 보장
    if ensure_cjk and any(has_cjk(c['keyword']) or has_hangul(c['keyword']) for c in picked):
        # 최소 1개는 CJK/Hangul 포함되도록 스왑
        if not any(has_cjk(c['keyword']) or has_hangul(c['keyword']) for c in final):
            # 첫 번째 CJK 후보를 끝자리에 교체
            for c in picked[top_n:]:
                if has_cjk(c['keyword']) or has_hangul(c['keyword']):
                    final[-1] = c
                    break

    # response_index 제거 (필요없음)
    final_keywords = [
        {'keyword': kw['keyword'], 'score': kw['score']}
        for kw in final
    ]
    
    return (conv_id, final_keywords)


def extract_all_keywords(
    grouped_responses: Dict[int, List[Dict[str, Any]]],
    conv_embeddings: Dict[int, np.ndarray],
    model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
    top_n: int = 5,
    keywords_per_response: int = 5,
    cache_dir: str = None,
    debug_conv_id: Optional[int] = None,
    debug_print_length: int = 200,
    ensure_cjk: bool = False
) -> Dict[int, tuple]:
    """
    모든 conversation에서 키워드 추출
    
    Returns:
        {conversation_id: keywords}
    """
    print("=" * 80)
    print("Step 3: 키워드 추출")
    print("=" * 80)
    
    print(f"KeyBERT 모델 로딩: {model_name}")
    if cache_dir:
        print(f"캐시 디렉토리: {cache_dir}")
        st_model = SentenceTransformer(model_name, cache_folder=cache_dir)
        kw_model = KeyBERT(model=st_model)
    else:
        kw_model = KeyBERT(model=model_name)
    
    results = {}
    
    print(f"\n총 {len(grouped_responses)}개 conversation 처리 중...")
    
    for conv_id, responses in tqdm(grouped_responses.items(), desc="키워드 추출"):
        conv_embedding = conv_embeddings.get(conv_id)
        
        conv_id, keywords = extract_keywords_for_conversation(
            conv_id, responses, conv_embedding, kw_model, top_n, keywords_per_response,
            debug_conv_id=debug_conv_id,
            debug_print_length=debug_print_length,
            ensure_cjk=ensure_cjk
        )
        
        results[conv_id] = keywords
    
    # 통계 출력
    total_with_keywords = sum(1 for kws in results.values() if len(kws) > 0)
    total_failed = len(results) - total_with_keywords
    
    print(f"\n" + "=" * 80)
    print(f"키워드 추출 완료:")
    print(f"  - 성공: {total_with_keywords}개 conversation")
    print(f"  - 실패: {total_failed}개 conversation")
    print("=" * 80 + "\n")
    
    return results


# ============================================================================
# 4. 결과 저장
# ============================================================================

def save_results(
    responses: List[Dict[str, Any]],
    keyword_results: Dict[int, tuple],
    output_path: str
):
    """Conversation 단위로 키워드를 저장"""
    print("=" * 80)
    print("Step 4: 결과 저장")
    print("=" * 80)
    
    # Conversation별로 그룹화하여 정보 수집
    conv_data = {}
    
    for resp in responses:
        conv_id = resp.get('conversation_id')
        
        if conv_id not in conv_data:
            conv_data[conv_id] = {
                'conversation_id': conv_id,
                'conversation_title': resp.get('conversation_title', ''),
                'response_count': 0,
                'keywords': []
            }
        
        conv_data[conv_id]['response_count'] += 1
        
        # 키워드 결과가 있으면 추가
        if conv_id in keyword_results:
            keywords = keyword_results[conv_id]
            conv_data[conv_id]['keywords'] = keywords
    
    # Conversation ID 순서로 정렬
    results = [conv_data[conv_id] for conv_id in sorted(conv_data.keys())]
    
    # JSON 저장
    print(f"저장 중: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"총 {len(results)}개 conversation 저장 완료\n")


def save_embeddings(conv_embeddings: Dict[int, np.ndarray], output_path: str):
    """임베딩을 PKL 파일로 저장"""
    items = []
    for conv_id, embedding in conv_embeddings.items():
        items.append({
            'conversation_id': conv_id,
            'embedding': embedding.tolist()
        })
    
    payload = {
        'items': items,
        'metadata': {
            'num_conversations': len(items),
            'embedding_dim': len(items[0]['embedding']) if items else 0,
            'method': 'mean_pooling'
        }
    }
    
    print(f"임베딩 저장 중: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"임베딩 저장 완료\n")


# ============================================================================
# 메인 파이프라인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="통합 키워드 추출 파이프라인")
    parser.add_argument("--input", type=str, required=True,
                       help="입력 AI 응답 JSON 파일 경로")
    parser.add_argument("--output", type=str, required=True,
                       help="출력 JSON 파일 경로 (키워드 포함)")
    parser.add_argument("--embedding-model", type=str,
                       default='paraphrase-multilingual-mpnet-base-v2',
                       help="(미사용) 임베딩 모델명 - 현재 mean pooling 비활성화")
    parser.add_argument("--keyword-model", type=str,
                       default='paraphrase-multilingual-mpnet-base-v2',
                       help="키워드 추출 모델명")
    parser.add_argument("--top-n", type=int, default=5,
                       help="conversation당 최종 선택할 키워드 수")
    parser.add_argument("--keywords-per-response", type=int, default=5,
                       help="각 응답마다 추출할 키워드 수 (기본: 5)")
    parser.add_argument("--debug-conv-id", type=int, default=None,
                       help="디버그할 conversation_id (전처리/토큰/후보 출력)")
    parser.add_argument("--debug-print-length", type=int, default=200,
                       help="디버그 시 출력할 전처리 텍스트 길이")
    parser.add_argument("--ensure-cjk", action="store_true",
                       help="가능하면 최종 키워드에 CJK/한글을 최소 1개 포함")
    parser.add_argument("--save-embeddings", type=str, default=None,
                       help="임베딩을 저장할 PKL 파일 경로 (선택)")
    parser.add_argument("--test-mode", action="store_true",
                       help="테스트 모드: 처음 10개 conversation만 처리")
    parser.add_argument("--cache-dir", type=str, default="models_cache",
                       help="모델 캐시 디렉토리 (기본: models_cache)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("통합 키워드 추출 파이프라인")
    print("=" * 80 + "\n")
    
    # Step 1: AI 응답 로딩
    responses = load_responses(args.input)
    grouped = group_by_conversation(responses)
    
    print(f"총 {len(grouped)}개 conversation")
    print(f"평균 응답 수: {len(responses) / len(grouped):.1f}\n")
    
    # 테스트 모드
    if args.test_mode:
        print("⚠️  테스트 모드: 처음 10개 conversation만 처리\n")
        conv_ids = list(grouped.keys())[:10]
        grouped = {k: v for k, v in grouped.items() if k in conv_ids}
        responses = [r for r in responses if r.get('conversation_id') in conv_ids]
    
    # Step 2: (비활성화) 임베딩 생성 스킵
    conv_embeddings = {}
    
    # Step 3: 키워드 추출
    keyword_results = extract_all_keywords(
        grouped,
        conv_embeddings,
        model_name=args.keyword_model,
        top_n=args.top_n,
        keywords_per_response=args.keywords_per_response,
        cache_dir=args.cache_dir,
        debug_conv_id=args.debug_conv_id,
        debug_print_length=args.debug_print_length,
        ensure_cjk=args.ensure_cjk
    )
    
    # Step 4: 결과 저장
    save_results(responses, keyword_results, args.output)
    
    # 임베딩 저장 (비활성화)
    
    # 완료
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"파이프라인 완료! (소요 시간: {elapsed:.1f}초)")
    print(f"결과 파일: {args.output}")
    if args.save_embeddings:
        print(f"임베딩 파일: {args.save_embeddings}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
