"""
S2 단계 테스트: 의미 분석 (SBERT + 개념 추출)
"""

import sys
from pathlib import Path
import json
import pickle
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.semantic_analyzer import SemanticAnalyzer
from analyze.parser import Note

def load_notes_from_s1():
    """S1 단계에서 생성한 노트 로드"""
    output_dir = project_root / "test" / "output"
    notes_file = output_dir / "s1_notes.json"

    if not notes_file.exists():
        raise FileNotFoundError(
            f"S1 결과 파일이 없습니다: {notes_file}\n"
            "먼저 test_s1.py를 실행하세요."
        )

    with open(notes_file, 'r', encoding='utf-8') as f:
        notes_dict = json.load(f)

    # dict를 Note 객체로 변환
    notes = [
        Note(
            note_id=n['note_id'],
            title=n['title'],
            content=n['content'],
            create_time=n['create_time'],
            update_time=n.get('update_time'),
            message_count=n.get('message_count', 0)
        )
        for n in notes_dict
    ]

    return notes


def test_s2():
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S2 단계 테스트: 의미 분석")
    print("=" * 80)

    # ============================================================
    # 1. S1 결과 로드
    # ============================================================
    print("\n[1] S1 결과 로드")
    print("-" * 80)

    load_start = time.time()
    notes = load_notes_from_s1()
    times['loading'] = time.time() - load_start
    print(f"✓ {len(notes)}개의 노트 로드 완료 ({times['loading']:.2f}초)\n")

    for i, note in enumerate(notes[:3], 1):
        print(f"노트 {i}: {note.title} ({len(note.content)} 글자)")

    # ============================================================
    # 2. 의미 분석 (SBERT + 개념 추출)
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 의미 분석 시작")
    print("-" * 80)

    model_start = time.time()
    analyzer = SemanticAnalyzer(
        model_name="jhgan/ko-sroberta-multitask",
        use_keybert=True,  # KeyBERT 활성화
        use_dbpedia=False
    )
    times['model_loading'] = time.time() - model_start

    # 노트 분석
    analysis_start = time.time()
    analysis_results = analyzer.analyze_notes(notes)
    times['analysis'] = time.time() - analysis_start

    print(f"\n✓ {len(analysis_results)}개의 노트 분석 완료 ({times['analysis']:.2f}초)\n")

    # 결과 확인
    print("분석 결과 샘플 (처음 3개):")
    for note_id, result in list(analysis_results.items())[:3]:
        print(f"\n노트 ID {note_id}:")
        print(f"  제목: {result['title']}")
        print(f"  임베딩 shape: {result['embedding'].shape}")
        print(f"  추출된 개념 ({len(result['concepts'])}개):")
        print(f"    {result['concepts'][:10]}")

    # ============================================================
    # 3. 유사도 계산
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 유사도 매트릭스 계산")
    print("-" * 80)

    similarity_start = time.time()
    embeddings_dict = {nid: res['embedding'] for nid, res in analysis_results.items()}
    similarities = analyzer.compute_similarity_matrix(embeddings_dict)
    times['similarity'] = time.time() - similarity_start

    print(f"✓ {len(similarities)}개의 유사도 쌍 계산 완료 ({times['similarity']:.2f}초)\n")

    # 상위 유사도 쌍 출력
    top_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
    print("상위 10개 유사도 쌍:")
    for i, ((n1, n2), sim) in enumerate(top_similarities, 1):
        title1 = analysis_results[n1]['title']
        title2 = analysis_results[n2]['title']
        print(f"  {i}. 노트 {n1} <-> 노트 {n2}: {sim:.4f}")
        print(f"     - {title1}")
        print(f"     - {title2}")

    # ============================================================
    # 4. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 결과 저장")
    print("-" * 80)

    save_start = time.time()
    output_dir = project_root / "test" / "output"

    # 1) 임베딩 저장 (pickle)
    embeddings_file = output_dir / "s2_embeddings.pkl"
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"✓ 임베딩 저장: {embeddings_file}")

    # 2) 개념 저장 (JSON)
    concepts_dict = {nid: res['concepts'] for nid, res in analysis_results.items()}
    concepts_file = output_dir / "s2_concepts.json"
    with open(concepts_file, 'w', encoding='utf-8') as f:
        json.dump(concepts_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ 개념 저장: {concepts_file}")

    # 3) 유사도 저장 (JSON)
    # tuple key를 string으로 변환
    similarities_serializable = {f"{k[0]},{k[1]}": v for k, v in similarities.items()}
    similarities_file = output_dir / "s2_similarities.json"
    with open(similarities_file, 'w', encoding='utf-8') as f:
        json.dump(similarities_serializable, f, indent=2)
    print(f"✓ 유사도 저장: {similarities_file}")

    # 4) 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    stats = {
        "total_notes": len(analysis_results),
        "embedding_dimension": list(embeddings_dict.values())[0].shape[0] if embeddings_dict else 0,
        "total_similarities": len(similarities),
        "avg_similarity": sum(similarities.values()) / len(similarities) if similarities else 0,
        "max_similarity": max(similarities.values()) if similarities else 0,
        "min_similarity": min(similarities.values()) if similarities else 0,
        "avg_concepts_per_note": sum(len(c) for c in concepts_dict.values()) / len(concepts_dict) if concepts_dict else 0,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "model_loading_seconds": round(times['model_loading'], 2),
            "analysis_seconds": round(times['analysis'], 2),
            "similarity_seconds": round(times['similarity'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s2_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ 통계 저장: {stats_file}")

    print("\n통계 요약:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("S2 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 데이터 로딩: {times['loading']:.2f}초")
    print(f"  - 모델 로딩: {times['model_loading']:.2f}초")
    print(f"  - 의미 분석: {times['analysis']:.2f}초")
    print(f"  - 유사도 계산: {times['similarity']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return analysis_results, similarities


if __name__ == "__main__":
    try:
        analysis_results, similarities = test_s2()
        print("\n✅ S2 테스트 성공!")
        print(f"분석된 노트: {len(analysis_results)}개")
        print(f"계산된 유사도 쌍: {len(similarities)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
