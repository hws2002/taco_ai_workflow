"""
메인 실행 스크립트
전체 파이프라인 실행: S1 -> S2 -> S3 -> S4
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
from analyze.semantic_analyzer import SemanticAnalyzer
from analyze.relationship_builder import RelationshipBuilder
from analyze.graph_storage import Neo4jStorage

# .env 파일 로드
load_dotenv()


def main(
    sample_size: int = None,
    similarity_threshold: float = 0.7,
    use_spacy: bool = True,
    use_conceptnet: bool = False,
    save_to_neo4j: bool = True,
    clear_db: bool = False
):
    """
    전체 파이프라인 실행

    Args:
        sample_size: 샘플 데이터 크기 (None이면 전체)
        similarity_threshold: 유사도 임계값 (0~1)
        use_spacy: spaCy 사용 여부
        use_conceptnet: ConceptNet 사용 여부
        save_to_neo4j: Neo4j에 저장 여부
        clear_db: Neo4j DB 초기화 여부
    """
    print("=" * 80)
    print("대화 데이터 지식 그래프 생성 시스템")
    print("=" * 80)

    # ============================================================
    # S1: 데이터 로딩 및 파싱
    # ============================================================
    print("\n[S1] 데이터 로딩 및 파싱")
    print("-" * 80)

    loader = ConversationLoader()

    if sample_size:
        print(f"샘플 데이터 {sample_size}개 로딩 중...")
        conversations = loader.load_sample(n=sample_size)
    else:
        print("전체 데이터 로딩 중...")
        conversations = loader.load()

    parser = NoteParser(min_content_length=20)
    notes = parser.parse_conversations(conversations)

    print(f"✓ {len(notes)}개의 노트 생성 완료")

    # ============================================================
    # S2: 의미 분석
    # ============================================================
    print("\n[S2] 의미 분석")
    print("-" * 80)

    analyzer = SemanticAnalyzer(
        model_name="jhgan/ko-sroberta-multitask",
        use_spacy=use_spacy,
        use_dbpedia=False
    )

    analysis_results = analyzer.analyze_notes(notes)
    embeddings_dict = {nid: res['embedding'] for nid, res in analysis_results.items()}
    concepts_dict = {nid: res['concepts'] for nid, res in analysis_results.items()}

    print(f"✓ {len(analysis_results)}개의 노트 분석 완료")

    # 유사도 계산
    similarities = analyzer.compute_similarity_matrix(embeddings_dict)
    print(f"✓ {len(similarities)}개의 유사도 쌍 계산 완료")

    # ============================================================
    # S3: 관계 추론
    # ============================================================
    print("\n[S3] 관계 추론")
    print("-" * 80)

    builder = RelationshipBuilder(
        similarity_threshold=similarity_threshold,
        use_conceptnet=use_conceptnet
    )

    if use_conceptnet:
        edges = builder.build_relationships(similarities, concepts_dict)
    else:
        edges = builder.build_simple_relationships(similarities)

    print(f"✓ {len(edges)}개의 엣지 생성 완료")

    # ============================================================
    # S4: Neo4j 저장
    # ============================================================
    if save_to_neo4j:
        print("\n[S4] Neo4j 저장")
        print("-" * 80)

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            with Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password) as storage:
                if clear_db:
                    print("\n데이터베이스 초기화 중...")
                    storage.clear_database()

                storage.create_constraints()
                storage.save_notes(notes)
                storage.save_edges(edges)
                storage.save_concepts(concepts_dict)

                # 통계 출력
                stats = storage.get_graph_stats()
                print("\n✓ Neo4j 저장 완료")
                print("\n그래프 통계:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"\n❌ Neo4j 저장 실패: {e}")
            print("Neo4j가 실행 중인지 확인하고 .env 파일의 연결 정보를 확인하세요.")

    # ============================================================
    # 완료
    # ============================================================
    print("\n" + "=" * 80)
    print("전체 파이프라인 완료!")
    print("=" * 80)

    print("\n요약:")
    print(f"  - 노트: {len(notes)}개")
    print(f"  - 엣지: {len(edges)}개")
    print(f"  - 개념: {sum(len(c) for c in concepts_dict.values())}개")

    if save_to_neo4j:
        print("\nNeo4j Browser에서 확인:")
        print("  http://localhost:7474")
        print("\n유용한 쿼리:")
        print("  MATCH (n:Note) RETURN n LIMIT 25")
        print("  MATCH (n:Note)-[r:RELATED_TO]->(m:Note) WHERE r.weight > 0.8 RETURN n, r, m LIMIT 50")
        print("  MATCH (n:Note)-[:HAS_CONCEPT]->(c:Concept) RETURN n, c LIMIT 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="대화 데이터 지식 그래프 생성")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="샘플 데이터 크기 (기본: 전체 데이터)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="유사도 임계값 (0~1, 기본: 0.7)"
    )

    parser.add_argument(
        "--no-spacy",
        action="store_true",
        help="spaCy 사용 안 함"
    )

    parser.add_argument(
        "--use-conceptnet",
        action="store_true",
        help="ConceptNet 사용 (느림)"
    )

    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Neo4j 저장 안 함"
    )

    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Neo4j DB 초기화"
    )

    args = parser.parse_args()

    try:
        main(
            sample_size=args.sample_size,
            similarity_threshold=args.threshold,
            use_spacy=not args.no_spacy,
            use_conceptnet=args.use_conceptnet,
            save_to_neo4j=not args.no_neo4j,
            clear_db=args.clear_db
        )

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
