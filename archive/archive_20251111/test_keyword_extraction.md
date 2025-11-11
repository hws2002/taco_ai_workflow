# 키워드 추출 테스트 가이드

## 문제점
- 현재 키워드 추출이 제대로 되지 않음
- 중국어의 경우 띄어쓰기가 없어 토크나이징이 어려움
- 다국어 처리 개선 필요

## 수정 내용

### 1. 다국어 토크나이징 개선
- **중국어**: jieba 라이브러리를 사용한 중국어 단어 분리
- **한국어/영어**: 기본 토크나이저 사용
- **언어 감지**: langdetect를 사용한 자동 언어 감지

### 2. Conversation 단위 처리
- 각 conversation의 모든 응답을 결합
- 결합된 텍스트의 임베딩 벡터를 mean pooling
- KeyBERT로 conversation당 상위 키워드 추출 (ngram 1-3)

### 3. 테스트 모드 추가
- `--test-mode` 플래그로 테스트 모드 활성화
- 영어, 한국어, 중국어 각각 50개 conversation 샘플링
- 빠른 테스트 및 결과 확인 가능

## 설치 필요 라이브러리

```bash
pip install jieba langdetect
```

## 테스트 실행 방법

### 1. 테스트 모드로 실행 (각 언어별 50개 conversation)

```bash
python extract_keywords_fast.py \
    --input test/output_full/s1_ai_responses.json \
    --output test/output_full/s2_ai_responses_with_keywords_test.json \
    --embeddings test/output_full/s2_conversation_embeddings_paraphrase-multilingual-mpnet-base-v2.pkl \
    --test-mode \
    --top-n 5 \
    --diversity 0.7 \
    --processes 4
```

### 2. 전체 데이터로 실행

```bash
python extract_keywords_fast.py \
    --input test/output_full/s1_ai_responses.json \
    --output test/output_full/s2_ai_responses_with_keywords.json \
    --embeddings test/output_full/s2_conversation_embeddings_paraphrase-multilingual-mpnet-base-v2.pkl \
    --top-n 5 \
    --diversity 0.7 \
    --processes 4
```

### 3. Conversation embedding이 없는 경우 (KeyBERT가 직접 계산)

```bash
python extract_keywords_fast.py \
    --input test/output_full/s1_ai_responses.json \
    --output test/output_full/s2_ai_responses_with_keywords.json \
    --no-precomputed-embeddings \
    --test-mode \
    --top-n 5 \
    --diversity 0.7
```

## 주요 파라미터 설명

- `--input`: AI 응답 JSON 파일 경로
- `--output`: 결과 저장 경로
- `--embeddings`: Conversation embedding PKL 파일 경로 (optional)
- `--test-mode`: 테스트 모드 활성화 (각 언어별 샘플링)
- `--samples-per-language`: 테스트 모드에서 언어별 샘플 수 (기본: 50)
- `--top-n`: Conversation당 추출할 키워드 수 (기본: 10)
- `--diversity`: MMR 다양성 파라미터 0-1 (기본: 0.7)
- `--nr-candidates`: 후보 키워드 수 (기본: 30)
- `--processes`: 병렬 처리 프로세스 수
- `--no-precomputed-embeddings`: 미리 계산된 embedding 사용 안 함

## 결과 확인

결과 JSON 파일에는 각 응답에 다음 정보가 추가됩니다:

```json
{
  "response_id": "1_0",
  "conversation_id": 1,
  "conversation_title": "제목",
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
```

## 예상 개선 사항

1. **중국어 키워드**: jieba 토크나이저로 단어 단위 분리 → 의미있는 키워드 추출
2. **한국어 키워드**: 기본 토크나이저로 어절/단어 추출 가능
3. **영어 키워드**: 기존과 동일하게 정상 작동
4. **언어 정보**: 각 conversation의 언어가 자동 감지되어 저장됨

## 트러블슈팅

### jieba 설치 실패 시
- Windows: `pip install jieba`
- Linux/Mac: `pip3 install jieba`

### langdetect 설치 실패 시
- `pip install langdetect`
- 설치가 안 되면 언어 감지 없이 진행 (경고만 출력)

### 메모리 부족 시
- `--processes 2` 등으로 프로세스 수 줄이기
- `--content-max-chars 2000` 등으로 텍스트 길이 제한

### 속도가 느린 경우
- `--no-precomputed-embeddings` 제거 (embedding 사용)
- `--processes` 값을 CPU 코어 수에 맞게 조정
