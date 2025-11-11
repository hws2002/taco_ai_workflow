"""
그래프 DB 저장 모듈 (S4)
Neo4j에 노드와 엣지를 저장
"""

from typing import List, Dict, Any, Optional
from .parser import Note
from .relationship_builder import Edge

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("경고: neo4j 드라이버가 설치되어 있지 않습니다.")
    print("설치: pip install neo4j")


class Neo4jStorage:
    """Neo4j 그래프 DB 저장 클래스"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        Args:
            uri: Neo4j URI
            user: 사용자 이름
            password: 비밀번호
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j 드라이버가 설치되어 있지 않습니다.")

        self.uri = uri
        self.user = user
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"✓ Neo4j 연결 성공: {uri}")
            self._verify_connection()
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            raise

    def _verify_connection(self):
        """연결 확인"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                record = result.single()
                if record and record["num"] == 1:
                    print("✓ Neo4j 연결 확인 완료")
        except Exception as e:
            print(f"❌ 연결 확인 실패: {e}")
            raise

    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j 연결 종료")

    def clear_database(self):
        """데이터베이스 초기화 (모든 노드와 관계 삭제)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ 데이터베이스 초기화 완료")

    def create_constraints(self):
        """제약 조건 생성 (노트 ID 유니크)"""
        with self.driver.session() as session:
            try:
                # Neo4j 버전에 따라 다른 문법
                session.run(
                    "CREATE CONSTRAINT note_id_unique IF NOT EXISTS "
                    "FOR (n:Note) REQUIRE n.id IS UNIQUE"
                )
                print("✓ 제약 조건 생성 완료")
            except Exception as e:
                # 이미 존재하거나 구버전인 경우
                print(f"제약 조건 생성 건너뜀: {e}")

    def save_notes(self, notes: List[Note]):
        """
        노트를 Neo4j에 저장

        Args:
            notes: Note 객체 리스트
        """
        print(f"\n{len(notes)}개의 노트 저장 중...")

        with self.driver.session() as session:
            for note in notes:
                session.run(
                    """
                    MERGE (n:Note {id: $id})
                    SET n.title = $title,
                        n.content = $content,
                        n.create_time = $create_time,
                        n.update_time = $update_time,
                        n.message_count = $message_count,
                        n.content_length = $content_length
                    """,
                    id=note.note_id,
                    title=note.title,
                    content=note.content,
                    create_time=note.create_time,
                    update_time=note.update_time,
                    message_count=note.message_count,
                    content_length=len(note.content)
                )

        print(f"✓ {len(notes)}개의 노트 저장 완료")

    def save_edges(self, edges: List[Edge]):
        """
        엣지(관계)를 Neo4j에 저장

        Args:
            edges: Edge 객체 리스트
        """
        print(f"\n{len(edges)}개의 엣지 저장 중...")

        with self.driver.session() as session:
            for edge in edges:
                session.run(
                    """
                    MATCH (source:Note {id: $source_id})
                    MATCH (target:Note {id: $target_id})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    SET r.weight = $weight,
                        r.label = $label,
                        r.reason = $reason
                    """,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    weight=edge.weight,
                    label=edge.label,
                    reason=edge.reason
                )

        print(f"✓ {len(edges)}개의 엣지 저장 완료")

    def save_concepts(self, concepts_dict: Dict[int, List[str]]):
        """
        개념을 Neo4j에 저장 (노트와 개념 노드 연결)

        Args:
            concepts_dict: {note_id: [concepts]} 형태의 dict
        """
        print(f"\n개념 저장 중...")

        total_concepts = sum(len(concepts) for concepts in concepts_dict.values())
        print(f"총 {total_concepts}개의 개념 연결 생성 중...")

        with self.driver.session() as session:
            for note_id, concepts in concepts_dict.items():
                for concept in concepts:
                    # 개념 노드 생성 및 노트와 연결
                    session.run(
                        """
                        MATCH (n:Note {id: $note_id})
                        MERGE (c:Concept {name: $concept})
                        MERGE (n)-[r:HAS_CONCEPT]->(c)
                        """,
                        note_id=note_id,
                        concept=concept
                    )

        print(f"✓ 개념 저장 완료")

    def get_graph_stats(self) -> Dict[str, int]:
        """
        그래프 통계 조회

        Returns:
            통계 정보 dict
        """
        stats = {}

        with self.driver.session() as session:
            # 노트 수
            result = session.run("MATCH (n:Note) RETURN count(n) AS count")
            stats['notes'] = result.single()['count']

            # 엣지 수
            result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS count")
            stats['edges'] = result.single()['count']

            # 개념 수
            result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
            stats['concepts'] = result.single()['count']

            # HAS_CONCEPT 관계 수
            result = session.run("MATCH ()-[r:HAS_CONCEPT]->() RETURN count(r) AS count")
            stats['concept_relations'] = result.single()['count']

        return stats

    def query_similar_notes(self, note_id: int, min_weight: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        특정 노트와 유사한 노트 조회

        Args:
            note_id: 조회할 노트 ID
            min_weight: 최소 가중치
            limit: 최대 결과 수

        Returns:
            유사한 노트 리스트
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (source:Note {id: $note_id})-[r:RELATED_TO]->(target:Note)
                WHERE r.weight >= $min_weight
                RETURN target.id AS id,
                       target.title AS title,
                       r.weight AS weight,
                       r.label AS label
                ORDER BY r.weight DESC
                LIMIT $limit
                """,
                note_id=note_id,
                min_weight=min_weight,
                limit=limit
            )

            return [dict(record) for record in result]

    def query_notes_by_concept(self, concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        특정 개념을 포함하는 노트 조회

        Args:
            concept: 개념 이름
            limit: 최대 결과 수

        Returns:
            노트 리스트
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Note)-[:HAS_CONCEPT]->(c:Concept {name: $concept})
                RETURN n.id AS id,
                       n.title AS title,
                       n.content_length AS content_length
                LIMIT $limit
                """,
                concept=concept,
                limit=limit
            )

            return [dict(record) for record in result]

    def export_graph_for_visualization(self) -> Dict[str, Any]:
        """
        시각화를 위한 그래프 데이터 추출

        Returns:
            {
                'nodes': [{'id': 1, 'title': '...', ...}],
                'edges': [{'source': 1, 'target': 2, 'weight': 0.8}, ...]
            }
        """
        with self.driver.session() as session:
            # 노드 추출
            nodes_result = session.run(
                """
                MATCH (n:Note)
                RETURN n.id AS id,
                       n.title AS title,
                       n.content_length AS content_length,
                       n.message_count AS message_count
                """
            )
            nodes = [dict(record) for record in nodes_result]

            # 엣지 추출
            edges_result = session.run(
                """
                MATCH (source:Note)-[r:RELATED_TO]->(target:Note)
                RETURN source.id AS source,
                       target.id AS target,
                       r.weight AS weight,
                       r.label AS label
                """
            )
            edges = [dict(record) for record in edges_result]

        return {'nodes': nodes, 'edges': edges}

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
