"""
KeyBERT를 사용하여 s1_ai_responses.json의 각 AI 응답에서 키워드를 추출하는 스크립트
"""

import json
from keybert import KeyBERT
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import argparse

def load_ai_responses(file_path: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 AI 응답 데이터를 로드합니다.

    Args:
        file_path: JSON 파일 경로

    Returns:
        AI 응답 객체 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 프로세스 로컬 모델 보관용 전역 변수
_KW_MODEL = None

def _init_kw_model(model_name: str):
    global _KW_MODEL
    _KW_MODEL = KeyBERT(model=model_name)

def _extract_one(response: Dict[str, Any],
                keyphrase_ngram_range,
                top_n: int,
                nr_candidates: int,
                diversity: float,
                stop_words: str,
                content_max_chars: int) -> Dict[str, Any]:
    content = response['content']
    if content_max_chars is not None:
        content = content[:content_max_chars]

    keywords = _KW_MODEL.extract_keywords(
        content,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        use_mmr=True,
        diversity=diversity,
        nr_candidates=nr_candidates,
        top_n=top_n
    )

    result = response.copy()
    result['keywords'] = [
        {
            'keyword': kw,
            'score': float(score)
        }
        for kw, score in keywords
    ]
    return result

def extract_keywords_from_responses(
    responses: List[Dict[str, Any]],
    kw_model: KeyBERT,
    top_n: int = 5,
    use_maxsum: bool = True,
    nr_candidates: int = 20,
    diversity: float = 0.5,
    use_parallel: bool = True,
    processes: int = None,
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    keyphrase_ngram_range: tuple = (1, 1),
    stop_words: str = 'english',
    content_max_chars: int = 1200
) -> List[Dict[str, Any]]:
    """
    각 AI 응답에서 KeyBERT를 사용하여 키워드를 추출합니다.

    Args:
        responses: AI 응답 객체 리스트
        kw_model: KeyBERT 모델 인스턴스
        top_n: 추출할 키워드 개수 (기본값: 5)
        use_maxsum: MaxSum을 사용할지 여부 (MMR로 대체됨)
        nr_candidates: 후보 키워드 개수 (기본값: 20)
        diversity: 다양성 파라미터 (0-1, 높을수록 다양한 키워드) (기본값: 0.5)
        use_parallel: 병렬 처리 여부 (기본값: True)
        processes: 사용할 프로세스 수 (None이면 CPU 코어 - 1)
        model_name: 병렬 처리 시 워커가 로드할 모델 이름
        keyphrase_ngram_range: 키프레이즈 n-gram 범위 (기본값: (1,1))
        stop_words: 불용어 설정 (기본값: 'english')
        content_max_chars: 각 응답에서 사용할 최대 문자 수 (기본값: 1200)

    Returns:
        키워드 정보가 추가된 응답 객체 리스트
    """
    if use_parallel:
        procs = processes if processes is not None else max(1, cpu_count() - 1)
        with Pool(processes=procs, initializer=_init_kw_model, initargs=(model_name,)) as pool:
            func = partial(
                _extract_one,
                keyphrase_ngram_range=keyphrase_ngram_range,
                top_n=top_n,
                nr_candidates=nr_candidates,
                diversity=diversity,
                stop_words=stop_words,
                content_max_chars=content_max_chars,
            )
            results = pool.map(func, responses)
        return results
    else:
        results = []
        for response in responses:
            content = response['content']
            if content_max_chars is not None:
                content = content[:content_max_chars]
            keywords = kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                use_mmr=True,
                diversity=diversity,
                nr_candidates=nr_candidates,
                top_n=top_n
            )
            result = response.copy()
            result['keywords'] = [
                {
                    'keyword': kw,
                    'score': float(score)
                }
                for kw, score in keywords
            ]
            results.append(result)
        return results

def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    결과를 JSON 파일로 저장합니다.

    Args:
        results: 키워드 추출 결과
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def print_summary(results: List[Dict[str, Any]]):
    """
    추출된 키워드의 요약 정보를 출력합니다.

    Args:
        results: 키워드 추출 결과
    """
    print(f"\n{'='*80}")
    print("키워드 추출 결과 요약")
    print(f"{'='*80}\n")

    for result in results:
        print(f"Response ID: {result['response_id']}")
        print(f"Conversation: {result['conversation_title']}")
        print(f"Content: {result['content'][:80]}...")
        print(f"Keywords:")
        for kw_info in result['keywords']:
            print(f"  - {kw_info['keyword']}: {kw_info['score']:.4f}")
        print()

def main():
    # CLI 인자 설정
    parser = argparse.ArgumentParser(description="KeyBERT 키워드 추출")
    parser.add_argument("--input", type=str, default='test/output/s1_ai_responses.json', help="입력 responses JSON 경로")
    parser.add_argument("--output", type=str, default='test/output/s2_ai_responses_with_keywords.json', help="출력 JSON 경로")
    parser.add_argument("--processes", type=int, default=None, help="병렬 프로세스 수 (기본: CPU-1)")
    parser.add_argument("--no-parallel", action="store_true", help="병렬 처리 비활성화")
    parser.add_argument("--top-n", type=int, default=5, help="문서당 키워드 수")
    parser.add_argument("--nr-candidates", type=int, default=10, help="후보 키워드 수")
    parser.add_argument("--diversity", type=float, default=0.5, help="MMR 다양성(0-1)")
    parser.add_argument("--content-max-chars", type=int, default=1200, help="본문 최대 길이")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    print("AI 응답 데이터 로딩 중...")
    responses = load_ai_responses(input_file)
    print(f"총 {len(responses)}개의 응답을 로드했습니다.\n")

    print("KeyBERT 모델 초기화 중...")
    # KeyBERT 모델 초기화
    # 다국어 지원을 위해 paraphrase-multilingual-MiniLM-L12-v2 모델 사용
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    print("모델 초기화 완료.\n")

    print("키워드 추출 중...")
    # 키워드 추출
    t0 = time.time()
    results = extract_keywords_from_responses(
        responses,
        kw_model,
        top_n=args.top_n,
        use_maxsum=False,  # MaxSum 비활성화 (MMR 사용)
        nr_candidates=args.nr_candidates,
        diversity=args.diversity,
        use_parallel=not args.no_parallel,
        processes=args.processes,
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        content_max_chars=args.content_max_chars
    )
    elapsed = time.time() - t0
    print("키워드 추출 완료.\n")
    print(f"총 소요 시간: {elapsed:.2f}초, 문서당 평균: {elapsed/len(responses):.3f}초")

    # 결과 저장
    print(f"결과를 {output_file}에 저장 중...")
    save_results(results, output_file)
    print("저장 완료.\n")

    # 요약 정보 출력
    print_summary(results)

    print(f"\n{'='*80}")
    print(f"처리 완료! 총 {len(results)}개의 응답에서 키워드를 추출했습니다.")
    print(f"결과 파일: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
