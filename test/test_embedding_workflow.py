"""
전체 임베딩 워크플로우 테스트
AI 답변 추출 -> 임베딩 생성 -> BERTopic 클러스터링 -> 문서별 풀링 -> 유사도 계산
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
from analyze.semantic_analyzer import SemanticAnalyzer
from analyze.embedding_processor import EmbeddingProcessor


def main():
    print("=" * 80)
    print("AI 답변 기반 문서 임베딩 워크플로우 테스트")
    print("=" * 80)

    # ============================================================
    # STEP 1: 데이터 로딩
    # ============================================================
    print("\n[STEP 1] 데이터 로딩")
    print("-" * 80)

    loader = ConversationLoader()

    # 샘플 데이터로 테스트 (전체 데이터는 너무 클 수 있음)
    sample_size = 50  # 50개 대화로 테스트
    print(f"샘플 {sample_size}개 대화 로딩 중...")
    conversations = loader.load_sample(n=sample_size)

    # ============================================================
    # STEP 2: AI 답변만 추출
    # ============================================================
    print("\n[STEP 2] AI 답변만 추출")
    print("-" * 80)

    parser = NoteParser(min_content_length=20)
    ai_responses = parser.parse_ai_responses(conversations)

    print(f"✓ 총 {len(ai_responses)}개의 AI 답변 추출 완료")

    # 샘플 출력
    if ai_responses:
        print(f"\n샘플 AI 답변:")
        sample_resp = ai_responses[0]
        print(f"  ID: {sample_resp.response_id}")
        print(f"  대화 제목: {sample_resp.conversation_title}")
        print(f"  내용 (앞 100자): {sample_resp.content[:100]}...")

    # ============================================================
    # STEP 3: 임베딩 생성
    # ============================================================
    print("\n[STEP 3] 임베딩 벡터 생성")
    print("-" * 80)

    # 다국어 모델 사용 (CLAUDE.md에서 권장)
    # bge-m3 또는 jhgan/ko-sroberta-multitask
    analyzer = SemanticAnalyzer(
        model_name="bge-m3",  # 다국어 모델
        use_keybert=True  # 키워드 추출은 활성화
    )

    # AI 답변 분석 (임베딩 + 개념 추출)
    response_embeddings = analyzer.analyze_ai_responses(ai_responses)

    print(f"✓ 총 {len(response_embeddings)}개의 임베딩 생성 완료")

    # 임베딩 차원 확인
    first_embedding = list(response_embeddings.values())[0]['embedding']
    print(f"  임베딩 차원: {first_embedding.shape[0]}")

    # ============================================================
    # STEP 4: BERTopic 클러스터링
    # ============================================================
    print("\n[STEP 4] BERTopic을 사용한 클러스터링")
    print("-" * 80)

    processor = EmbeddingProcessor(
        min_topic_size=3,  # 작은 샘플이므로 최소 크기 줄임
        nr_topics=None,  # 자동 결정
        language="multilingual",
        verbose=True
    )

    # response_embeddings에 content 추가 (BERTopic이 문서 텍스트 필요)
    for response_id, emb_data in response_embeddings.items():
        # ai_responses에서 해당 response 찾기
        for resp in ai_responses:
            if resp.response_id == response_id:
                emb_data['content'] = resp.content
                break

    # BERTopic 클러스터링
    clustered_responses, topic_keywords = processor.cluster_with_bertopic(response_embeddings)

    print(f"\n✓ 클러스터링 완료")
    print(f"  생성된 토픽 수: {len(set(cr.topic_id for cr in clustered_responses.values()))}")

    # ============================================================
    # STEP 5: 문서별 풀링
    # ============================================================
    print("\n[STEP 5] 대화별 토픽 풀링")
    print("-" * 80)

    document_embeddings = processor.pool_by_conversation(
        clustered_responses,
        response_embeddings
    )

    print(f"✓ {len(document_embeddings)}개의 대화에 대한 주제 벡터 생성 완료")

    # 샘플 문서 정보 출력
    if document_embeddings:
        print(f"\n샘플 문서:")
        sample_conv_id = list(document_embeddings.keys())[0]
        sample_doc = document_embeddings[sample_conv_id]
        print(f"  대화 ID: {sample_doc.conversation_id}")
        print(f"  제목: {sample_doc.conversation_title}")
        print(f"  주제 수: {len(sample_doc.topic_embeddings)}")
        print(f"  주제 목록:")
        for topic_id, keywords in sample_doc.topic_keywords.items():
            keywords_str = ", ".join(keywords[:5])
            print(f"    토픽 {topic_id}: {keywords_str}")

    # ============================================================
    # STEP 6: 유사도 계산
    # ============================================================
    print("\n[STEP 6] 문서 간 유사도 계산")
    print("-" * 80)

    similarities = processor.compute_all_document_similarities(document_embeddings)

    print(f"\n✓ 유사도 계산 완료")

    # 가장 유사한 문서 쌍 10개 출력
    print(f"\n가장 유사한 문서 쌍 TOP 10:")
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for i, ((conv_id_1, conv_id_2), sim) in enumerate(sorted_sims[:10], 1):
        doc1 = document_embeddings[conv_id_1]
        doc2 = document_embeddings[conv_id_2]
        print(f"  {i}. 유사도 {sim:.4f}")
        print(f"     [{conv_id_1}] {doc1.conversation_title}")
        print(f"     [{conv_id_2}] {doc2.conversation_title}")

    # ============================================================
    # 완료
    # ============================================================
    print("\n" + "=" * 80)
    print("전체 워크플로우 테스트 완료!")
    print("=" * 80)

    print("\n요약:")
    print(f"  - 대화 수: {len(conversations)}")
    print(f"  - AI 답변 수: {len(ai_responses)}")
    print(f"  - 생성된 토픽 수: {len(topic_keywords)}")
    print(f"  - 문서별 평균 주제 수: {sum(len(d.topic_embeddings) for d in document_embeddings.values()) / len(document_embeddings):.2f}")
    print(f"  - 유사도 쌍 수: {len(similarities)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
