# 통합 키워드 추출 파이프라인

## 개요

AI 응답에서 키워드를 추출하는 완전 통합 파이프라인입니다.

### 처리 단계
1. **AI 응답 로딩** - JSON 파일에서 응답 데이터 로드
2. **임베딩 생성** - Conversation별로 응답을 평균 pooling하여 대표 벡터 생성
3. **키워드 추출** - KeyBERT로 각 conversation에서 상위 키워드 추출 (다국어 지원)
4. **결과 저장** - 키워드가 추가된 응답을 JSON으로 저장

### 주요 개선사항
- ✅ **단순화된 키워드 추출**: stop_words 제거로 안정성 향상
- ✅ **다국어 지원**: 중국어(jieba), 한국어, 영어 자동 감지 및 처리
- ✅ **통합 파이프라인**: 한 번의 실행으로 모든 단계 완료
- ✅ **오류 처리**: 각 conversation별로 독립적 처리, 실패해도 계속 진행

## 빠른 시작

### 1. 테스트 실행 (10개 conversation)

```bash
run_pipeline.bat
```

또는 직접 실행:

```bash
conda activate taco
pip install jieba langdetect

python pipeline_extract_keywords.py ^
    --input test/output_full/s1_ai_responses.json ^
    --output test/output_full/s2_keywords_pipeline_test.json ^
    --top-n 5 ^
    --test-mode
```

### 2. 전체 데이터 실행

```bash
run_pipeline_full.bat
```

또는 직접 실행:

```bash
conda activate taco

python pipeline_extract_keywords.py ^
    --input test/output_full/s1_ai_responses.json ^
    --output test/output_full/s2_ai_responses_with_keywords.json ^
    --save-embeddings test/output_full/conversation_embeddings.pkl ^
    --top-n 5
```

## 파라미터 설명

### 필수 파라미터
- `--input`: 입력 AI 응답 JSON 파일 경로
- `--output`: 출력 JSON 파일 경로 (키워드 포함)

### 선택 파라미터
- `--embedding-model`: 임베딩 모델명 (기본: `paraphrase-multilingual-mpnet-base-v2`)
- `--keyword-model`: 키워드 추출 모델명 (기본: `paraphrase-multilingual-MiniLM-L12-v2`)
- `--top-n`: Conversation당 추출할 키워드 수 (기본: 5)
- `--save-embeddings`: 임베딩을 저장할 PKL 파일 경로 (선택)
- `--test-mode`: 테스트 모드 (처음 10개 conversation만 처리)

## 출력 형식

### JSON 결과 파일

```json
[
  {
    "response_id": "1_0",
    "conversation_id": 1,
    "conversation_title": "제목",
    "message_index": 0,
    "content": "응답 내용...",
    "timestamp": 1234567890.123,
    "keywords": [
      {
        "keyword": "키워드1",
        "score": 0.85
      },
      {
        "keyword": "키워드2",
        "score": 0.72
      }
    ],
    "detected_language": "ko"
  }
]
```

### PKL 임베딩 파일 (선택)

```python
{
  'items': [
    {
      'conversation_id': 1,
      'embedding': [0.123, 0.456, ...]  # 768차원 벡터
    }
  ],
  'metadata': {
    'num_conversations': 100,
    'embedding_dim': 768,
    'method': 'mean_pooling'
  }
}
```

## 결과 확인

```bash
python check_keywords_result.py --file test/output_full/s2_ai_responses_with_keywords.json --samples 5
```

출력 예시:
```
================================================================================
키워드 추출 결과 분석
================================================================================

총 응답 수: 10000
총 conversation 수: 500

================================================================================
언어별 통계
================================================================================

[ko]
  - Conversation 수: 200
  - 평균 키워드 수: 4.8
  - 키워드 수 범위: 3-5

[en]
  - Conversation 수: 250
  - 평균 키워드 수: 4.9
  - 키워드 수 범위: 4-5

[zh-cn]
  - Conversation 수: 50
  - 평균 키워드 수: 4.5
  - 키워드 수 범위: 2-5
```

## 기술 스택

### 필수 라이브러리
- `sentence-transformers`: 임베딩 생성
- `keybert`: 키워드 추출
- `numpy`: 수치 계산
- `tqdm`: 진행률 표시

### 선택 라이브러리 (다국어 지원)
- `jieba`: 중국어 토크나이징
- `langdetect`: 언어 자동 감지

설치:
```bash
pip install sentence-transformers keybert numpy tqdm jieba langdetect
```

## 문제 해결

### jieba 미설치 경고
```bash
pip install jieba
```

### langdetect 미설치 경고
```bash
pip install langdetect
```

### 메모리 부족
- 배치 크기 줄이기 (코드 내 `batch_size` 수정)
- 테스트 모드로 일부만 처리 (`--test-mode`)

### 키워드가 추출되지 않음
1. 텍스트가 너무 짧은지 확인 (최소 20자 필요)
2. 코드 블록/특수문자만 있는지 확인
3. 로그에서 에러 메시지 확인

## 성능

### 예상 처리 속도
- 임베딩 생성: ~50 conversations/초
- 키워드 추출: ~2 conversations/초
- 전체 파이프라인: 1000 conversations ≈ 10-15분

### GPU 사용
sentence-transformers는 자동으로 GPU를 감지하고 사용합니다.
GPU가 없으면 CPU로 자동 전환됩니다.

## 다음 단계

1. **테스트 실행**: `run_pipeline.bat`으로 10개 테스트
2. **결과 확인**: 키워드 품질 검증
3. **전체 실행**: `run_pipeline_full.bat`으로 전체 데이터 처리
4. **후속 분석**: 추출된 키워드로 클러스터링, 토픽 모델링 등

## 기존 코드와의 차이점

### `extract_keywords_fast.py` (기존)
- ❌ 복잡한 병렬 처리
- ❌ stop_words 설정 문제
- ❌ 워커 프로세스에서 jieba 미설치 이슈

### `pipeline_extract_keywords.py` (신규)
- ✅ 단순한 순차 처리
- ✅ stop_words 비활성화로 안정성 향상
- ✅ 메인 프로세스에서 모든 라이브러리 사용
- ✅ 통합 파이프라인 (응답 → 임베딩 → 키워드)
