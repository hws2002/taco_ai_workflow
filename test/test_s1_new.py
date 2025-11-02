"""
S1 단계 테스트: 데이터 로딩 및 AI 답변 추출
(새로운 워크플로우: AI 답변 기반 문서 분류)
"""

import sys
from pathlib import Path
import time
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser


def test_s1_new(sample_size=50, use_cache=False):
    """
    S1 테스트: AI 답변 추출

    새 워크플로우:
    1. 데이터 로딩
    2. AI 답변만 추출 (사용자 질문 제외)
    3. 결과 저장
    """
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S1 단계 테스트: 데이터 로딩 및 AI 답변 추출")
    print("(새로운 워크플로우: AI 답변 기반)")
    print("=" * 80)

    # ============================================================
    # 1. 데이터 로딩
    # ============================================================
    print("\n[1] 데이터 로딩")
    print("-" * 80)

    load_start = time.time()
    loader = ConversationLoader()

    print(f"샘플 {sample_size}개 대화 로딩 중...")
    conversations = loader.load_sample(n=sample_size)
    times['loading'] = time.time() - load_start

    print(f"✓ {len(conversations)}개의 대화 로드 완료")
    print(f"⏱️  소요 시간: {times['loading']:.2f}초\n")

    # 대화 정보 확인
    print("처음 3개 대화 정보:")
    for i, conv in enumerate(conversations[:3], 1):
        info = loader.get_conversation_info(conv)
        print(f"\n대화 {i}:")
        print(f"  제목: {info['title']}")
        print(f"  생성시간: {info['create_time']}")
        print(f"  메시지 수: {info['message_count']}")

    # ============================================================
    # 2. AI 답변만 추출
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] AI 답변만 추출")
    print("-" * 80)

    parse_start = time.time()
    parser = NoteParser(min_content_length=20)

    # 새 워크플로우: AI 답변만 추출
    ai_responses = parser.parse_ai_responses(conversations)
    times['parsing'] = time.time() - parse_start

    print(f"\n✓ 총 {len(ai_responses)}개의 AI 답변 추출 완료")
    print(f"⏱️  소요 시간: {times['parsing']:.2f}초\n")

    # AI 답변 정보 출력
    print("추출된 AI 답변 정보:")
    for i, resp in enumerate(ai_responses[:5], 1):
        print(f"\n답변 {i}:")
        print(f"  ID: {resp.response_id}")
        print(f"  대화 제목: {resp.conversation_title}")
        print(f"  대화 ID: {resp.conversation_id}")
        print(f"  메시지 인덱스: {resp.message_index}")
        print(f"  내용 길이: {len(resp.content)} 글자")
        print(f"  내용 미리보기: {resp.content[:150]}...")

    # 통계 정보
    print("\n통계:")
    print(f"  - 대화당 평균 AI 답변 수: {len(ai_responses) / len(conversations):.1f}개")
    print(f"  - 평균 답변 길이: {sum(len(r.content) for r in ai_responses) / len(ai_responses):.0f} 글자")

    # ============================================================
    # 3. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 결과 저장")
    print("-" * 80)

    save_start = time.time()
    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # AI 답변을 dict로 변환
    responses_dict = []
    for resp in ai_responses:
        responses_dict.append({
            "response_id": resp.response_id,
            "conversation_id": resp.conversation_id,
            "conversation_title": resp.conversation_title,
            "message_index": resp.message_index,
            "content": resp.content,
            "timestamp": resp.timestamp
        })

    # JSON 저장
    output_file = output_dir / "s1_ai_responses.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ AI 답변 저장: {output_file}")
    print(f"  - 총 {len(responses_dict)}개 답변")
    print(f"  - 파일 크기: {output_file.stat().st_size / 1024:.2f} KB")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    stats = {
        "total_responses": len(ai_responses),
        "total_conversations": len(conversations),
        "avg_responses_per_conversation": len(ai_responses) / len(conversations) if conversations else 0,
        "avg_content_length": sum(len(r.content) for r in ai_responses) / len(ai_responses) if ai_responses else 0,
        "sample_size": sample_size,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "parsing_seconds": round(times['parsing'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s1_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 통계 저장: {stats_file}")

    print("\n" + "=" * 80)
    print("S1 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 데이터 로딩: {times['loading']:.2f}초")
    print(f"  - AI 답변 추출: {times['parsing']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return ai_responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S1 테스트: AI 답변 추출")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    args = parser.parse_args()

    try:
        ai_responses = test_s1_new(sample_size=args.sample_size)
        print(f"\n✅ S1 테스트 성공!")
        print(f"다음 단계에서 사용할 AI 답변: {len(ai_responses)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
