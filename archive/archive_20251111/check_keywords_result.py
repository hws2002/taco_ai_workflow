"""
키워드 추출 결과 확인 스크립트
각 언어별로 샘플을 확인하여 키워드가 제대로 추출되었는지 검증
"""

import json
import argparse
from collections import defaultdict

def check_results(filepath: str, samples_per_lang: int = 3):
    """
    키워드 추출 결과를 언어별로 분석하고 샘플 출력
    
    Args:
        filepath: 결과 JSON 파일 경로 (conversation 단위)
        samples_per_lang: 언어별 출력할 샘플 수
    """
    print("=" * 80)
    print("키워드 추출 결과 분석")
    print("=" * 80)
    
    # JSON 로드
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 결과 형식 확인
    if not results:
        print("결과가 없습니다.")
        return
    
    first_item = results[0]
    is_conversation_level = 'response_count' in first_item
    
    if is_conversation_level:
        print(f"\n✓ Conversation 단위 결과 형식")
        print(f"총 conversation 수: {len(results)}")
    else:
        print(f"\n✓ Response 단위 결과 형식")
        print(f"총 응답 수: {len(results)}")
        return
    
    conversations = results
    
    # 언어별 통계
    lang_stats = defaultdict(int)
    lang_keyword_counts = defaultdict(list)
    lang_samples = defaultdict(list)
    
    for item in conversations:
        conv_id = item.get('conversation_id')
        keywords = item.get('keywords', [])
        response_count = item.get('response_count', 0)
        title = item.get('conversation_title', 'N/A')
        
        # 키워드 개수별 통계
        kw_count = len(keywords)
        lang_stats['all'] += 1
        lang_keyword_counts['all'].append(kw_count)
        
        # 샘플 저장
        if len(lang_samples['all']) < samples_per_lang:
            lang_samples['all'].append({
                'conv_id': conv_id,
                'title': title,
                'keywords': keywords,
                'num_responses': response_count
            })
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("키워드 추출 통계")
    print("=" * 80)
    
    if 'all' in lang_stats:
        count = lang_stats['all']
        avg_kw = sum(lang_keyword_counts['all']) / max(1, len(lang_keyword_counts['all']))
        print(f"\n  - 총 Conversation 수: {count}")
        print(f"  - 평균 키워드 수: {avg_kw:.2f}")
        print(f"  - 키워드 수 범위: {min(lang_keyword_counts['all'])}-{max(lang_keyword_counts['all'])}")
    
    # 샘플 출력
    print("\n" + "=" * 80)
    print("키워드 추출 샘플")
    print("=" * 80)
    
    if 'all' in lang_samples:
        for i, sample in enumerate(lang_samples['all'], 1):
            print(f"\n[샘플 {i}]")
            print(f"Conversation ID: {sample['conv_id']}")
            print(f"제목: {sample['title'][:60]}...")
            print(f"응답 수: {sample['num_responses']}")
            print(f"키워드 ({len(sample['keywords'])}개):")
            
            for kw_info in sample['keywords']:
                keyword = kw_info['keyword']
                score = kw_info['score']
                print(f"  - {keyword}: {score:.4f}")
    
    # 키워드가 없는 conversation 체크
    print("\n" + "=" * 80)
    print("키워드 추출 실패 체크")
    print("=" * 80)
    
    no_keywords = []
    for item in conversations:
        keywords = item.get('keywords', [])
        if len(keywords) == 0:
            no_keywords.append((item.get('conversation_id'), item.get('conversation_title', 'N/A')))
    
    if no_keywords:
        print(f"\n키워드가 추출되지 않은 conversation: {len(no_keywords)}개")
        for conv_id, title in no_keywords[:5]:
            print(f"  - Conv {conv_id}: {title[:60]}")
        if len(no_keywords) > 5:
            print(f"  ... 외 {len(no_keywords) - 5}개")
    else:
        print("\n✓ 모든 conversation에서 키워드가 정상적으로 추출되었습니다!")
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="키워드 추출 결과 확인")
    parser.add_argument("--file", type=str, 
                       default='test/output_full/s2_ai_responses_with_keywords_test.json',
                       help="결과 JSON 파일 경로")
    parser.add_argument("--samples", type=int, default=3,
                       help="언어별 샘플 출력 수 (기본: 3)")
    
    args = parser.parse_args()
    
    try:
        check_results(args.file, args.samples)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {args.file}")
        print("\n먼저 키워드 추출을 실행해주세요:")
        print("  python extract_keywords_fast.py --test-mode ...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
