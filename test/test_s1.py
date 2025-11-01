"""
S1 단계 테스트: 데이터 로딩 및 파싱
"""

import sys
from pathlib import Path
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
import json

def test_s1(sample_size=10):
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S1 단계 테스트: 데이터 로딩 및 파싱")
    print("=" * 80)

    # ============================================================
    # 1. 데이터 로딩
    # ============================================================
    print("\n[1] 데이터 로딩")
    print("-" * 80)

    load_start = time.time()
    loader = ConversationLoader()

    # 샘플 데이터 로드
    print(f"샘플 데이터 {sample_size}개 로딩 중...")
    conversations = loader.load_sample(n=sample_size)
    times['loading'] = time.time() - load_start
    print(f"✓ {len(conversations)}개의 대화 로드 완료 ({times['loading']:.2f}초)\n")

    # 대화 정보 확인
    print("처음 3개 대화 정보:")
    for i, conv in enumerate(conversations[:3], 1):
        info = loader.get_conversation_info(conv)
        print(f"\n대화 {i}:")
        print(f"  제목: {info['title']}")
        print(f"  생성시간: {info['create_time']}")
        print(f"  업데이트: {info['update_time']}")
        print(f"  메시지 수: {info['message_count']}")

    # ============================================================
    # 2. 파싱
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 노트 파싱")
    print("-" * 80)

    parse_start = time.time()
    parser = NoteParser(min_content_length=20)
    notes = parser.parse_conversations(conversations)
    times['parsing'] = time.time() - parse_start

    print(f"\n✓ {len(notes)}개의 노트 생성 완료 ({times['parsing']:.2f}초)\n")

    # 노트 정보 출력
    print("생성된 노트 정보:")
    for i, note in enumerate(notes[:5], 1):
        print(f"\n노트 {i}:")
        print(f"  ID: {note.note_id}")
        print(f"  제목: {note.title}")
        print(f"  메시지 수: {note.message_count}")
        print(f"  내용 길이: {len(note.content)} 글자")
        print(f"  내용 미리보기: {note.content[:150]}...")

    # ============================================================
    # 3. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 결과 저장")
    print("-" * 80)

    save_start = time.time()
    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # 노트를 dict로 변환
    notes_dict = parser.export_notes_to_dict(notes)

    # JSON 저장
    output_file = output_dir / "s1_notes.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notes_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ 노트 저장: {output_file}")
    print(f"  - 총 {len(notes_dict)}개 노트")
    print(f"  - 파일 크기: {output_file.stat().st_size / 1024:.2f} KB")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    stats = {
        "total_notes": len(notes),
        "total_conversations": len(conversations),
        "avg_content_length": sum(len(n.content) for n in notes) / len(notes) if notes else 0,
        "avg_message_count": sum(n.message_count for n in notes) / len(notes) if notes else 0,
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
    print(f"  - 노트 파싱: {times['parsing']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return notes


if __name__ == "__main__":
    try:
        notes = test_s1(sample_size=30)
        print("\n✅ S1 테스트 성공!")
        print(f"다음 단계에서 사용할 노트: {len(notes)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
