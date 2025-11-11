"""
KeyBERT를 사용하여 conversation 단위로 키워드를 추출하는 최적화된 스크립트
- 텍스트 전처리 (이모티콘, 코드 블록 제거)
- conversation 단위 처리
- 속도 최적화
"""

import json
import re
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import argparse
import os
import pickle
import numpy as np
import sys
import random

# Windows console encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 중국어 토크나이저 임포트 (optional)
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not installed. Chinese tokenization will be suboptimal.")
    print("Install with: pip install jieba")

# 언어 감지 임포트 (optional)
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Language detection disabled.")
    print("Install with: pip install langdetect")

def detect_language(text: str) -> str:
    """
    텍스트의 언어를 감지
    
    Args:
        text: 입력 텍스트
    
    Returns:
        언어 코드 ('en', 'ko', 'zh-cn', 'ja' 등) 또는 'unknown'
    """
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
        return 'unknown'
    
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'


def tokenize_chinese(text: str) -> str:
    """
    중국어 텍스트를 jieba로 토크나이징하여 띄어쓰기 추가
    
    Args:
        text: 중국어 텍스트
    
    Returns:
        띄어쓰기가 추가된 텍스트
    """
    if not JIEBA_AVAILABLE:
        return text
    
    # jieba로 분리 후 띄어쓰기로 조인
    words = jieba.cut(text)
    return ' '.join(words)


def clean_text(text: str, language: Optional[str] = None) -> str:
    """
    텍스트 전처리: 이모티콘, 코드 블록 등 제거

    Args:
        text: 원본 텍스트
        language: 언어 코드 (None이면 자동 감지)

    Returns:
        정제된 텍스트
    """
    # 1. 코드 블록 제거 (```...```)
    text = re.sub(r'```[\s\S]*?```', ' ', text)

    # 2. 인라인 코드 제거 (`...`)
    text = re.sub(r'`[^`]+`', ' ', text)

    # 3. 이모티콘 제거 (Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FAFF"  # Chess Symbols
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)

    # 4. 특수 문자 중 일부만 제거 (보존: 알파벳, 숫자, 기본 문장부호, 한글, 중국어, 일본어)
    # URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 5. 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    # 6. 중국어인 경우 jieba로 토크나이징
    if language is None:
        language = detect_language(text)
    
    if language in ['zh-cn', 'zh-tw', 'zh'] and JIEBA_AVAILABLE:
        text = tokenize_chinese(text)

    return text


def load_ai_responses(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 AI 응답 데이터를 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_conversation_embeddings(embeddings_path: str) -> Dict[int, np.ndarray]:
    """
    Conversation embedding PKL 파일을 로드

    Args:
        embeddings_path: conversation embedding PKL 파일 경로

    Returns:
        conversation_id를 키로 하는 embedding 딕셔너리
    """
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    emb_dict = {}
    for item in data.get('items', []):
        conv_id = item.get('conversation_id')
        embedding = item.get('embedding')
        if conv_id is not None and embedding is not None:
            emb_dict[conv_id] = np.array(embedding)

    return emb_dict


def group_responses_by_conversation(responses: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    응답을 conversation_id별로 그룹화

    Args:
        responses: AI 응답 리스트

    Returns:
        conversation_id를 키로 하는 응답 그룹
    """
    grouped = {}
    for resp in responses:
        conv_id = resp.get('conversation_id')
        if conv_id not in grouped:
            grouped[conv_id] = []
        grouped[conv_id].append(resp)
    return grouped


# 프로세스 로컬 모델 및 embeddings
_KW_MODEL = None
_CONV_EMBEDDINGS = None

def _init_kw_model(conv_embeddings: Dict[int, np.ndarray], model_name: str, cache_dir: str = None):
    """
    워커 프로세스 초기화

    Args:
        conv_embeddings: conversation_id를 키로 하는 embedding 딕셔너리
        model_name: SentenceTransformer 모델명
        cache_dir: 모델 캐시 디렉터리
    """
    from sentence_transformers import SentenceTransformer
    global _KW_MODEL, _CONV_EMBEDDINGS
    _CONV_EMBEDDINGS = conv_embeddings
    # KeyBERT 모델 초기화 (후보 키워드 embedding용)
    st_model = SentenceTransformer(model_name, cache_folder=cache_dir) if cache_dir else SentenceTransformer(model_name)
    _KW_MODEL = KeyBERT(model=st_model)


def create_custom_vectorizer(language: str = 'unknown') -> CountVectorizer:
    """
    언어별 커스텀 CountVectorizer 생성
    
    Args:
        language: 언어 코드
    
    Returns:
        CountVectorizer 인스턴스
    """
    if language in ['zh-cn', 'zh-tw', 'zh'] and JIEBA_AVAILABLE:
        # 중국어: jieba 토크나이저 사용
        def jieba_tokenizer(text):
            return list(jieba.cut(text))
        return CountVectorizer(tokenizer=jieba_tokenizer, ngram_range=(1, 3))
    else:
        # 기타 언어: 기본 토크나이저 사용
        return CountVectorizer(ngram_range=(1, 3))


def _extract_conversation_keywords(
    conv_data: tuple,
    top_n: int,
    nr_candidates: int,
    diversity: float,
    stop_words: str,
    content_max_chars: int,
    use_precomputed_embeddings: bool
) -> List[Dict[str, Any]]:
    """
    하나의 conversation에서 키워드 추출

    Args:
        conv_data: (conversation_id, responses_list)
        기타 KeyBERT 파라미터들
        use_precomputed_embeddings: True면 미리 계산된 conversation embedding 사용,
                                     False면 KeyBERT가 현장에서 계산

    Returns:
        키워드가 추가된 응답 리스트
    """
    conv_id, responses = conv_data

    # 1. 모든 응답 텍스트 결합 및 언어 감지
    combined_text = ""
    for resp in responses:
        combined_text += resp['content'] + " "
    
    # 언어 감지
    language = detect_language(combined_text)
    
    # 2. 언어별 전처리
    cleaned_text = ""
    for resp in responses:
        cleaned = clean_text(resp['content'], language=language)
        cleaned_text += cleaned + " "

    # 3. 길이 제한
    if content_max_chars and len(cleaned_text) > content_max_chars:
        cleaned_text = cleaned_text[:content_max_chars]

    # 4. conversation 전체에서 키워드 추출
    try:
        # doc_embeddings 준비
        doc_embedding = None
        if use_precomputed_embeddings and _CONV_EMBEDDINGS:
            doc_embedding = _CONV_EMBEDDINGS.get(conv_id)
            # KeyBERT는 2D embedding (1, dim)을 기대하므로 reshape
            if doc_embedding is not None and len(doc_embedding.shape) == 1:
                doc_embedding = doc_embedding.reshape(1, -1)

        # 언어별 vectorizer 생성
        vectorizer = create_custom_vectorizer(language)
        
        keywords = _KW_MODEL.extract_keywords(
            cleaned_text,
            keyphrase_ngram_range=(1, 3),  # unigram + bigram + trigram
            stop_words=stop_words,
            use_mmr=True,
            diversity=diversity,
            nr_candidates=nr_candidates,
            top_n=top_n,
            doc_embeddings=doc_embedding,  # None이면 KeyBERT가 자체 계산
            vectorizer=vectorizer  # 커스텀 vectorizer 사용
        )
    except Exception as e:
        print(f"Warning: Failed to extract keywords for conversation {conv_id}: {e}")
        keywords = []

    # 5. 키워드를 각 응답에 할당 (동일한 키워드 세트)
    kw_list = [
        {
            'keyword': kw,
            'score': float(score)
        }
        for kw, score in keywords
    ]

    results = []
    for resp in responses:
        result = resp.copy()
        result['keywords'] = kw_list
        result['detected_language'] = language  # 언어 정보 추가
        results.append(result)

    return results


def sample_responses_by_language(
    responses: List[Dict[str, Any]],
    samples_per_language: int = 50
) -> List[Dict[str, Any]]:
    """
    각 언어별로 conversation을 샘플링
    
    Args:
        responses: AI 응답 리스트
        samples_per_language: 각 언어별 샘플링할 conversation 수
    
    Returns:
        샘플링된 응답 리스트
    """
    # conversation별로 그룹화
    grouped = group_responses_by_conversation(responses)
    
    # 각 conversation의 언어 감지
    conv_languages = {}
    print("언어 감지 중...")
    for conv_id, conv_responses in tqdm(grouped.items(), desc="Language detection"):
        combined_text = " ".join([r['content'][:200] for r in conv_responses[:3]])  # 처음 3개 응답만
        lang = detect_language(combined_text)
        conv_languages[conv_id] = lang
    
    # 언어별로 conversation 분류
    lang_convs = {}
    for conv_id, lang in conv_languages.items():
        if lang not in lang_convs:
            lang_convs[lang] = []
        lang_convs[lang].append(conv_id)
    
    # 언어별 통계 출력
    print(f"\n언어별 conversation 분포:")
    for lang, conv_ids in sorted(lang_convs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {lang}: {len(conv_ids)}개")
    
    # 주요 언어 (영어, 한국어, 중국어)별로 샘플링
    target_languages = {
        'en': samples_per_language,
        'ko': samples_per_language,
        'zh-cn': samples_per_language,
        'zh': samples_per_language
    }
    
    sampled_conv_ids = set()
    for lang, sample_count in target_languages.items():
        if lang in lang_convs:
            available = lang_convs[lang]
            sample_size = min(sample_count, len(available))
            sampled = random.sample(available, sample_size)
            sampled_conv_ids.update(sampled)
            print(f"  샘플링: {lang} -> {sample_size}개 conversation")
    
    # 샘플링된 conversation의 응답만 추출
    sampled_responses = [r for r in responses if r.get('conversation_id') in sampled_conv_ids]
    
    print(f"\n총 {len(sampled_conv_ids)}개 conversation, {len(sampled_responses)}개 응답 샘플링됨")
    
    return sampled_responses


def extract_keywords_by_conversation(
    responses: List[Dict[str, Any]],
    conv_embeddings: Dict[int, np.ndarray],
    model_name: str,
    top_n: int = 10,
    nr_candidates: int = 30,
    diversity: float = 0.7,
    use_parallel: bool = True,
    processes: int = None,
    stop_words: str = None,
    content_max_chars: int = 3000,
    cache_dir: str = None,
    use_precomputed_embeddings: bool = True
) -> List[Dict[str, Any]]:
    """
    conversation 단위로 키워드 추출

    Args:
        responses: AI 응답 리스트
        conv_embeddings: conversation_id를 키로 하는 embedding 딕셔너리 (None 가능)
        model_name: 후보 키워드 embedding을 위한 SentenceTransformer 모델명
        top_n: conversation당 추출할 키워드 수 (기본값: 10)
        nr_candidates: 후보 키워드 수 (기본값: 30)
        diversity: 다양성 파라미터 (기본값: 0.7)
        use_parallel: 병렬 처리 여부
        processes: 프로세스 수
        stop_words: 불용어
        content_max_chars: conversation당 최대 문자 수 (기본값: 3000)
        cache_dir: 모델 캐시 디렉터리
        use_precomputed_embeddings: True면 conv_embeddings 사용, False면 KeyBERT가 자체 계산

    Returns:
        키워드가 추가된 응답 리스트
    """
    # conversation별로 그룹화
    grouped = group_responses_by_conversation(responses)
    conv_items = list(grouped.items())

    print(f"총 {len(responses)}개 응답을 {len(conv_items)}개 conversation으로 그룹화")

    all_results = []

    if use_parallel:
        procs = processes if processes is not None else max(1, cpu_count() - 1)
        with Pool(processes=procs, initializer=_init_kw_model, initargs=(conv_embeddings, model_name, cache_dir)) as pool:
            func = partial(
                _extract_conversation_keywords,
                top_n=top_n,
                nr_candidates=nr_candidates,
                diversity=diversity,
                stop_words=stop_words,
                content_max_chars=content_max_chars,
                use_precomputed_embeddings=use_precomputed_embeddings
            )
            with tqdm(total=len(conv_items), desc="키워드 추출 (conversation)", unit="conv") as pbar:
                for conv_results in pool.imap_unordered(func, conv_items):
                    all_results.extend(conv_results)
                    pbar.update(1)
    else:
        # 비병렬 처리
        global _KW_MODEL, _CONV_EMBEDDINGS
        from sentence_transformers import SentenceTransformer
        _CONV_EMBEDDINGS = conv_embeddings
        st_model = SentenceTransformer(model_name, cache_folder=cache_dir) if cache_dir else SentenceTransformer(model_name)
        _KW_MODEL = KeyBERT(model=st_model)

        with tqdm(total=len(conv_items), desc="키워드 추출 (conversation)", unit="conv") as pbar:
            for conv_data in conv_items:
                conv_results = _extract_conversation_keywords(
                    conv_data,
                    top_n=top_n,
                    nr_candidates=nr_candidates,
                    diversity=diversity,
                    stop_words=stop_words,
                    content_max_chars=content_max_chars,
                    use_precomputed_embeddings=use_precomputed_embeddings
                )
                all_results.extend(conv_results)
                pbar.update(1)

    # response_id 기준으로 정렬하여 원래 순서 유지
    result_dict = {r['response_id']: r for r in all_results}
    ordered_results = [result_dict[resp['response_id']] for resp in responses if resp['response_id'] in result_dict]

    return ordered_results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """결과를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results: List[Dict[str, Any]], max_display: int = 5):
    """추출된 키워드의 요약 정보 출력"""
    print(f"\n{'='*80}")
    print("키워드 추출 결과 요약")
    print(f"{'='*80}\n")

    # conversation별로 그룹화해서 표시
    grouped = group_responses_by_conversation(results)

    for i, (conv_id, responses) in enumerate(list(grouped.items())[:max_display]):
        first_resp = responses[0]
        print(f"Conversation ID: {conv_id}")
        print(f"Title: {first_resp.get('conversation_title', 'N/A')}")
        print(f"응답 수: {len(responses)}")
        print(f"Keywords:")
        for kw_info in first_resp.get('keywords', [])[:10]:
            print(f"  - {kw_info['keyword']}: {kw_info['score']:.4f}")
        print()

    if len(grouped) > max_display:
        print(f"... 외 {len(grouped) - max_display}개 conversation")


def main():
    parser = argparse.ArgumentParser(description="Conversation 단위 고속 키워드 추출")
    parser.add_argument("--input", type=str, default='test/output/s1_ai_responses.json', help="입력 responses JSON 경로")
    parser.add_argument("--embeddings", type=str, default=None, help="Conversation embedding PKL 파일 경로 (optional)")
    parser.add_argument("--output", type=str, default='test/output/s2_ai_responses_with_keywords.json', help="출력 JSON 경로")
    parser.add_argument("--processes", type=int, default=None, help="병렬 프로세스 수 (기본: CPU-1)")
    parser.add_argument("--no-parallel", action="store_true", help="병렬 처리 비활성화")
    parser.add_argument("--top-n", type=int, default=10, help="conversation당 키워드 수 (기본: 10)")
    parser.add_argument("--nr-candidates", type=int, default=30, help="후보 키워드 수 (기본: 30)")
    parser.add_argument("--diversity", type=float, default=0.7, help="MMR 다양성 (기본: 0.7)")
    parser.add_argument("--content-max-chars", type=int, default=3000, help="conversation당 최대 문자 수 (기본: 3000)")
    parser.add_argument("--kw-model", type=str, default='paraphrase-multilingual-MiniLM-L12-v2', help="후보 키워드 embedding을 위한 KeyBERT 모델명")
    parser.add_argument("--kw-cache-dir", type=str, default='models_cache', help="모델 캐시 디렉터리")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")
    parser.add_argument("--no-precomputed-embeddings", action="store_true", help="미리 계산된 embedding 사용 안 함 (KeyBERT가 현장에서 계산)")
    parser.add_argument("--test-mode", action="store_true", help="테스트 모드: 각 언어별로 50개 conversation 샘플링")
    parser.add_argument("--samples-per-language", type=int, default=50, help="테스트 모드에서 각 언어별 샘플링 수 (기본: 50)")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    print("AI 응답 데이터 로딩 중...")
    responses = load_ai_responses(input_file)
    print(f"총 {len(responses)}개의 응답을 로드했습니다.\n")
    
    # 테스트 모드: 언어별 샘플링
    if args.test_mode:
        print("\n=== 테스트 모드: 언어별 샘플링 ===\n")
        responses = sample_responses_by_language(responses, args.samples_per_language)
        print()

    # Conversation embedding 로딩 (optional)
    conv_embeddings = {}
    use_precomputed = not args.no_precomputed_embeddings and args.embeddings is not None

    if use_precomputed:
        print("Conversation embedding 로딩 중...")
        conv_embeddings = load_conversation_embeddings(args.embeddings)
        print(f"총 {len(conv_embeddings)}개 conversation의 embedding을 로드했습니다.\n")
    else:
        print("미리 계산된 embedding 사용 안 함 - KeyBERT가 현장에서 계산합니다.\n")

    # 캐시 체크
    cache_path = output_file
    existing_by_id = {}
    if not args.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for item in existing:
                rid = item.get('response_id')
                if rid is not None:
                    existing_by_id[rid] = item
            print(f"캐시에서 {len(existing_by_id)}개 응답 발견")
        except Exception:
            existing_by_id = {}

    missing_responses = [r for r in responses if r.get('response_id') not in existing_by_id]
    print(f"신규 처리 필요: {len(missing_responses)}개 응답")

    new_results = []
    if missing_responses:
        print("\nKeyBERT 모델 초기화 중...")
        print("키워드 추출 시작...\n")

        t0 = time.time()
        new_results = extract_keywords_by_conversation(
            missing_responses,
            conv_embeddings=conv_embeddings,
            model_name=args.kw_model,
            top_n=args.top_n,
            nr_candidates=args.nr_candidates,
            diversity=args.diversity,
            use_parallel=not args.no_parallel,
            processes=args.processes,
            stop_words=None,  # 다국어 처리를 위해 None 사용
            content_max_chars=args.content_max_chars,
            cache_dir=args.kw_cache_dir if not args.no_cache else None,
            use_precomputed_embeddings=use_precomputed
        )
        elapsed = time.time() - t0

        # conversation 개수 계산
        conv_ids = set(r['conversation_id'] for r in missing_responses)
        num_convs = len(conv_ids)

        print("\n키워드 추출 완료!")
        print(f"총 소요 시간: {elapsed:.2f}초")
        print(f"conversation당 평균: {elapsed/max(1,num_convs):.3f}초")
        print(f"응답당 평균: {elapsed/max(1,len(missing_responses)):.3f}초")

    # 결과 병합
    new_by_id = {item['response_id']: item for item in new_results}
    merged = []
    for r in responses:
        rid = r.get('response_id')
        if rid in existing_by_id:
            merged.append(existing_by_id[rid])
        elif rid in new_by_id:
            merged.append(new_by_id[rid])
        else:
            tmp = r.copy()
            tmp['keywords'] = []
            merged.append(tmp)

    # 저장
    print(f"\n결과를 {output_file}에 저장 중...")
    save_results(merged, output_file)
    print("저장 완료.\n")

    # 요약
    print_summary(merged)

    print(f"\n{'='*80}")
    print(f"처리 완료! 총 {len(merged)}개 응답 결과 저장")
    print(f"  - 신규: {len(new_results)}개")
    print(f"  - 캐시: {len(merged) - len(new_results)}개")
    print(f"결과 파일: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
