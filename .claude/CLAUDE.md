# 다음과같은 파이프라인에 해당하는 코드를 작성해야 함.


**INPUT**: json형식의 대화내용(질문/ 대화 내용 전체 미정)

**OUTPUT**: 

1. Node, Edge 정보를 포함한 json
2. 메타데이터
    1. 어느 대분류 cluster에 포함하는지
    2. (미정)

# STEP0 : 문서 embedding vector, key word 추출

## 불용어 처리

라이브러리 가져와서 

영어/중국어/한국어

## 문서 embedding vector 추출

유저가 여러 언어를 섞어서 질문하는 상황에 대비해 다국어 버전 모델을 사용하기로 함.

- SBERT
    - mBERT
- 구글 Universal Sentence Encoder
- 미정(?)

| 모델 계열 | 정확도 | 학습 속도 | 추론 속도 | 사용 권장 상황 |
| --- | --- | --- | --- | --- |
| **SBERT 계열** | ⭐⭐⭐⭐⭐ (최상) | 느림 | 매우 빠름 | 정확도가 가장 중요하고, 의미 기반의 정교한 분석이 필요할 때 |
| **USE** | ⭐⭐⭐⭐ (높음) | 보통 | 빠름 | 범용적이고 안정적인 성능이 필요할 때 |
| **Doc2Vec** | ⭐⭐ (보통) | 빠름 | 해당 없음 | SBERT가 부담스럽지만, TF-IDF보다는 의미를 반영하고 싶을 때 |

**고려사항**

1. 각 모델들의 토큰 제한
    1. 유저의 채팅 길이를 제한하거나
    2. 채팅을 잘 compress해서 모델에 먹여야 함.

## keyword 추출

- KeyBert
- spaCy
- BERTopic

| **Feature** | **KeyBERT** | **spaCy** | **BERTopic** |
| --- | --- | --- | --- |
| **Main Goal** | Single-doc keyword extraction | General NLP tasks | Multi-doc topic discovery |
| **Method** | Semantic Similarity (BERT) | Rule-based (Grammar) | Topic Modeling (BERT + Clustering) |
| **Accuracy** | **Very High** ⭐⭐⭐⭐⭐ | Low to Medium ⭐⭐ | **High (for Topics)** ⭐⭐⭐⭐ |
| **Speed** | Slow to Medium ⭐⭐ | **Very Fast** ⭐⭐⭐⭐⭐ | Slow ⭐ |
| **Use Case** | Get best keywords from one text. | Fast entity/noun extraction. | Find themes in many texts. |

고려사항: 

1. 키워드가 얼마나 잘 뽑히는지
2. 각 키워드 점수 가중치 어떻게 배정되는지

# STEP1-1 : 대분류 (clustering)(G3)

추출한 키워드를 기반으로 대분류를 진행

### 고려중인 방식들

1. LLM 사용 (유력)
2. 전통적인 clustering 방식들
    1. K-means
    2. HDBSCAN
    3. etc…

| **기법** | **핵심 원리** | **클러스터 형태** | **K 지정 필요 (클러스터 개수)** | **이상치 처리** | **적합한 상황** |
| --- | --- | --- | --- | --- | --- |
| **K-Means** | **중심점(Centroid) 기반** 분할 | 원형(Spherical) | **필요** | 처리 못함 | 데이터가 구형으로 잘 분리되고, 빠르고 간단한 분석이 필요할 때 |
| **계층적** | **거리 기반** 병합/분할 | 제약 없음 | **불필요** | 처리 못함 | 데이터의 계층 구조를 시각적으로 파악하고 싶을 때 (소규모 데이터) |
| **DBSCAN** | **밀도(Density) 기반** 확장 | 임의의 형태 | **불필요** | **매우 잘함** | 복잡한 모양의 군집을 찾거나, 데이터의 이상치를 제거하고 싶을 때 |

# STEP2: 문서간 유사도 계산(G1)

1. cosine 유사도 (유력)
2. jaccard 거리
3. 등..

등을 통해 문서간 유사도를 계산

유사도를 기준으로 각기 다른 edge 생성 방식이 사용된다.

# STEP2-1 : threshold2 ≤ 유사도

두 문서간 edge를 확정적으로 생성한다.  이 문서들간의 관계는 추후에 지식 그래프를 활용해서 더욱 명확한 관계유형을 추론해낼 것임.

**관계유형예시**

Causality

- /r/Causes
- /r/CapableOf
- /r/MotivatedByGoal

 Equivalency

- /r/Synonym
- /r/SimilarTo

Opposition

- /r/Antonym
- /r/DistinctFrom

Dependency

- /r/HasPrerequisite
- /r/HasContext
- /r/HasProperty
- /r/PartOf

 General

- /r/IsA
- /r/RelatedTo

# STEP2-2 : threshold1 ≤ 유사도 ≤ threshold2

해당 구간에 포함하는 edge들에 대해, 또 다른 metrics를 사용해 (tf-idf 혹은 doc2vec cosine 유사도) threshold3을 넘으면 LLM에게 edge생성여부를 문의.

<aside>
💡

LLM 사용 비중을 최대한 합리적으로 하기 위해 적절한 threshold를 선택해야 함.

추후 여러 실험을 통해 확정해 나갈 것.

</aside>

<aside>
💡

Q : LLM을 왜 여기서 사용하나요?

A : 현재 단계에서 원하는 것은 유저가 알아채지 못한 혹시 모를 “insight” 발견을 위함임. 오히려 threshold2를 넘지 못한 edge를 LLM에게 물어봄으로서, 유저도 “확실하게” 알 수 있는 link가 아닌, 잠재적인 link를 찾아낼 수 있도록 LLM을 사용하는 것.

</aside>

# 추가 STEP(G2, 미정)

STEP2 결과 바탕으로 클러스터링(or 다른 방법으로) 이용 → 중분류(G2) 생성

---

# 문서 분류

이때, 문서 임베딩 추출에 해당하는 워크플로우는 다음과 같음.

1.  **데이터 선택:** 노이즈가 많은 사용자 질문 대신, 잘 정제된 **AI의 답변만을 사용**한다.

2. **임베딩 벡터추출과 클러스터링**: 
[출력 1] 먼저 SentenceTransformer (예: bge-m3)를 이용해 모든 AI 답변(문서)의 임베딩 벡터를 직접 만듭니다.

이 벡터들을 BERTopic에 전달하여 학습시킵니다.

[출력 3] BERTopic은 이 벡터들을 클러스터링하여 각 문서에 **클러스터 ID(토픽 ID)**를 할당합니다.

[출력 2] BERTopic은 각 클러스터 ID가 어떤 의미인지 **대표 키워드(토픽)**를 뽑아줍니다.


4.  **문서 내 풀링:** 이제 다시 개별 채팅 세션으로 돌아와, 그 안에 속한 답변 벡터들을 **할당된 클러스터 ID별로 그룹핑**하고, 같은 그룹 내 벡터들을 **풀링(평균)**한다.
    * 결과: 하나의 채팅 세션은 1개가 아닌, **여러 개의 '주제 벡터'**를 갖게 됩니다. (예: 채팅 A = {날씨 주제 벡터, 맛집 추천 벡터})
5.  **유사도 계산:** 두 채팅 세션(A, B)을 비교할 때, 각 세션이 가진 **주제 벡터 집합 간의 평균 거리**를 계산하여 최종 유사도로 삼는다.

6. 이때, 코사인 유사도가 0.7이상인 채팅간에는 무조건적으로 엣지를 생성하고, 0.5이상, 0.7미만인 경우에는 일단 보류. (단 따로 저장해 둘것)

## LLM대분류 스텝
7. 이후, STEP1-1에 따라 대분류를 진행할것
각 채팅 기록들의 키워드 목록을 LLM에 전달한 후, LLM에게 그것을 기반으로, 3~5개의 대분류를 진행해달라고 요청.

8. 이후, 아까 유사도를 기반으로 한 엣지중에, 총 엣지 개수, 각 클러스터안에서의 노드만으로 생성된 엣지의 수, 대분류 클러스터를 넘어서서 생성된 엣지의 수에 대한 통계를 낼 것.

9. 생성된 엣지와 노드를 기반으로, 간단한 그래프 뷰를 제공할 것. 이떄, LLM대분류에 따라 다른 색깔을 제공해서 생성