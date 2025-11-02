"""
KeyBERT를 사용하여 s1_ai_responses.json의 각 AI 응답에서 키워드를 추출하는 스크립트
"""

import json
from keybert import KeyBERT
from typing import List, Dict, Any

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

def extract_keywords_from_responses(
    responses: List[Dict[str, Any]],
    kw_model: KeyBERT,
    top_n: int = 5,
    use_maxsum: bool = True,
    nr_candidates: int = 20,
    diversity: float = 0.5
) -> List[Dict[str, Any]]:
    """
    각 AI 응답에서 KeyBERT를 사용하여 키워드를 추출합니다.

    Args:
        responses: AI 응답 객체 리스트
        kw_model: KeyBERT 모델 인스턴스
        top_n: 추출할 키워드 개수 (기본값: 5)
        use_maxsum: MaxSum을 사용할지 여부 (기본값: True)
        nr_candidates: 후보 키워드 개수 (기본값: 20)
        diversity: 다양성 파라미터 (0-1, 높을수록 다양한 키워드) (기본값: 0.5)

    Returns:
        키워드 정보가 추가된 응답 객체 리스트
    """
    results = []

    for response in responses:
        content = response['content']

        # KeyBERT를 사용하여 키워드 추출
        if use_maxsum:
            # MaxSum 방식: 다양성을 고려한 키워드 추출
            keywords = kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 2),  # 1-2개 단어로 구성된 키워드
                stop_words='english',
                use_maxsum=True,
                nr_candidates=nr_candidates,
                top_n=top_n,
                diversity=diversity
            )
        else:
            # 기본 방식: 코사인 유사도 기반
            keywords = kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )

        # 결과에 키워드 정보 추가
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
    # 파일 경로 설정
    input_file = 'test/output/s1_ai_responses.json'
    output_file = 'test/output/s2_ai_responses_with_keywords.json'

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
    results = extract_keywords_from_responses(
        responses,
        kw_model,
        top_n=5,           # 각 문서당 5개의 키워드 추출
        use_maxsum=True,   # 다양성을 고려한 MaxSum 방식 사용
        nr_candidates=20,  # 20개의 후보 중에서 선택
        diversity=0.5      # 중간 수준의 다양성
    )
    print("키워드 추출 완료.\n")

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
