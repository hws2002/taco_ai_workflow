"""
전체 테스트 단계 시간 종합 리포트
"""

import sys
from pathlib import Path
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_stats():
    """각 단계의 통계 파일 로드"""
    output_dir = project_root / "test" / "output"

    stats = {}
    for stage in ['s1', 's2', 's3']:
        stats_file = output_dir / f"{stage}_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats[stage] = json.load(f)
        else:
            stats[stage] = None

    return stats


def print_summary(stats):
    """종합 리포트 출력"""
    print("=" * 80)
    print("TACO 파이프라인 실행 시간 종합 리포트")
    print("=" * 80)

    total_time = 0
    stage_names = {
        's1': 'S1: 데이터 로딩 및 파싱',
        's2': 'S2: 의미 분석',
        's3': 'S3: 관계 추론'
    }

    # 각 단계별 시간 출력
    for stage in ['s1', 's2', 's3']:
        print(f"\n{'─' * 80}")
        print(f"📊 {stage_names[stage]}")
        print('─' * 80)

        if stats[stage] is None:
            print(f"  ⚠️  {stage}_stats.json 파일을 찾을 수 없습니다.")
            continue

        if 'elapsed_time' not in stats[stage]:
            print(f"  ⚠️  시간 정보가 없습니다. 테스트를 다시 실행해주세요.")
            continue

        elapsed = stats[stage]['elapsed_time']

        # 세부 시간 출력
        for key, value in elapsed.items():
            if key != 'total_seconds':
                label = key.replace('_seconds', '').replace('_', ' ').title()
                print(f"  • {label:20s}: {value:8.2f}초")

        # 전체 시간
        stage_total = elapsed['total_seconds']
        print(f"  {'─' * 40}")
        print(f"  • {'Total':20s}: {stage_total:8.2f}초")
        total_time += stage_total

    # 전체 파이프라인 시간
    print("\n" + "=" * 80)
    print("⏱️  전체 파이프라인 실행 시간")
    print("=" * 80)
    print(f"  총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")

    # 각 단계별 비율
    if total_time > 0:
        print("\n  단계별 비율:")
        for stage in ['s1', 's2', 's3']:
            if stats[stage] and 'elapsed_time' in stats[stage]:
                stage_time = stats[stage]['elapsed_time']['total_seconds']
                percentage = (stage_time / total_time) * 100
                bar_length = int(percentage / 2)
                bar = '█' * bar_length
                print(f"    {stage_names[stage]:30s}: {bar:50s} {percentage:5.1f}%")

    # 데이터 통계
    print("\n" + "=" * 80)
    print("📈 처리 데이터 통계")
    print("=" * 80)

    if stats['s1']:
        print(f"  • 총 대화 수: {stats['s1'].get('total_conversations', 'N/A')}")
        print(f"  • 생성된 노트 수: {stats['s1'].get('total_notes', 'N/A')}")

    if stats['s2']:
        print(f"  • 임베딩 차원: {stats['s2'].get('embedding_dimension', 'N/A')}")
        print(f"  • 계산된 유사도 쌍: {stats['s2'].get('total_similarities', 'N/A')}")
        print(f"  • 평균 유사도: {stats['s2'].get('avg_similarity', 0):.4f}")

    if stats['s3']:
        print(f"  • 생성된 엣지 수: {stats['s3'].get('total_edges', 'N/A')}")
        print(f"  • 평균 엣지 가중치: {stats['s3'].get('avg_weight', 0):.4f}")

    print("\n" + "=" * 80)


def save_summary_report(stats):
    """종합 리포트를 JSON 파일로 저장"""
    output_dir = project_root / "test" / "output"

    total_time = sum(
        stats[stage]['elapsed_time']['total_seconds']
        for stage in ['s1', 's2', 's3']
        if stats[stage] and 'elapsed_time' in stats[stage]
    )

    report = {
        "total_pipeline_seconds": round(total_time, 2),
        "total_pipeline_minutes": round(total_time / 60, 2),
        "stages": {}
    }

    for stage in ['s1', 's2', 's3']:
        if stats[stage] and 'elapsed_time' in stats[stage]:
            report['stages'][stage] = {
                "elapsed_time": stats[stage]['elapsed_time'],
                "percentage": round((stats[stage]['elapsed_time']['total_seconds'] / total_time) * 100, 2) if total_time > 0 else 0
            }

    report_file = output_dir / "pipeline_summary.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 종합 리포트 저장: {report_file}\n")


if __name__ == "__main__":
    try:
        stats = load_stats()
        print_summary(stats)
        save_summary_report(stats)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
