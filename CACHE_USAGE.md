# 캐싱 기능 사용 가이드

임베딩 계산과 BERTopic 클러스터링은 시간이 오래 걸리는 작업입니다.
같은 데이터로 반복 실험할 때는 **캐싱 기능**을 사용하면 처리 속도를 크게 향상시킬 수 있습니다.

## 📦 재사용 가능한 것들

### 1. **SBERT 모델 파일** (자동 캐싱)
- **위치**: `models/models--BAAI--bge-m3/`
- **크기**: 약 2.27GB
- **재사용**: 한 번 다운로드 후 자동으로 재사용
- **삭제 금지**: 이 폴더를 삭제하면 다시 다운로드해야 함

### 2. **임베딩 벡터** (수동 캐싱)
- **위치**: `cache/response_embeddings.pkl`
- **내용**: 820개 AI 답변의 1024차원 임베딩 벡터
- **재사용**: `--use-cache` 옵션으로 로드

### 3. **클러스터링 결과** (수동 캐싱)
- **위치**: `cache/clustered_responses.pkl`
- **내용**: BERTopic 클러스터링 결과 + 토픽 키워드
- **재사용**: `--use-cache` 옵션으로 로드

### 4. **문서별 풀링 결과** (수동 캐싱)
- **위치**: `cache/document_embeddings.pkl`
- **내용**: 대화별로 풀링된 주제 벡터
- **재사용**: `--use-cache` 옵션으로 로드

### 5. **유사도 결과** (수동 캐싱)
- **위치**: `cache/similarities.pkl`
- **내용**: 모든 문서 쌍 간의 유사도
- **재사용**: `--use-cache` 옵션으로 로드

## 🚀 사용 방법

### 첫 번째 실행 (캐시 생성)
```bash
# 캐시를 저장하면서 실행
python -X utf8 test/test_embedding_workflow.py --use-cache

# 또는 샘플 크기 지정
python -X utf8 test/test_embedding_workflow.py --use-cache --sample-size 100
```

**실행 시간**: 약 5~10분 (모델 다운로드 + 임베딩 계산 + 클러스터링)

### 두 번째 실행부터 (캐시 재사용)
```bash
# 캐시 재사용 - 매우 빠름!
python -X utf8 test/test_embedding_workflow.py --use-cache
```

**실행 시간**: 약 10~30초 (캐시에서 로드만)

### 캐시 초기화
```bash
# 캐시 삭제 후 처음부터 다시 계산
python -X utf8 test/test_embedding_workflow.py --use-cache --clear-cache
```

### 캐시 없이 실행
```bash
# 캐시 사용 안 함 (매번 새로 계산)
python -X utf8 test/test_embedding_workflow.py
```

## 📊 성능 비교

| 실행 방식 | 실행 시간 | 설명 |
|---------|---------|------|
| **첫 실행 (캐시 생성)** | 5~10분 | 모델 다운로드 + 모든 계산 + 캐시 저장 |
| **캐시 재사용** | 10~30초 | 캐시에서 로드만 (🚀 20~60배 빠름) |
| **캐시 없이 실행** | 3~5분 | 모델은 재사용하지만 매번 계산 |

## 💡 사용 시나리오

### 시나리오 1: 파라미터 튜닝
BERTopic의 `min_topic_size`나 `n_clusters` 같은 파라미터만 바꾸고 싶을 때:

```bash
# 1단계: 임베딩만 캐시에 저장 (첫 실행)
python -X utf8 test/test_embedding_workflow.py --use-cache

# 2단계: embedding_processor.py의 파라미터 수정

# 3단계: 임베딩은 재사용, 클러스터링만 다시 실행
# (cache/clustered_responses.pkl 삭제)
rm cache/clustered_responses.pkl
python -X utf8 test/test_embedding_workflow.py --use-cache
```

### 시나리오 2: 데이터 변경
새로운 대화 데이터를 추가했을 때:

```bash
# 캐시 초기화 후 처음부터 다시
python -X utf8 test/test_embedding_workflow.py --use-cache --clear-cache
```

### 시나리오 3: 빠른 테스트
코드 수정 후 빠르게 결과만 확인하고 싶을 때:

```bash
# 캐시 재사용 (10초 안에 결과 확인)
python -X utf8 test/test_embedding_workflow.py --use-cache
```

## 📁 캐시 디렉토리 구조

```
cache/
├── response_embeddings.pkl      # AI 답변 임베딩 (가장 큼, 수십 MB)
├── clustered_responses.pkl      # 클러스터링 결과
├── document_embeddings.pkl      # 문서별 풀링 결과
└── similarities.pkl             # 유사도 결과
```

## 🧹 캐시 관리

### 캐시 파일 확인
```bash
ls -lh cache/
```

### 캐시 파일 삭제
```bash
# 전체 삭제
rm -rf cache/

# 특정 파일만 삭제 (예: 클러스터링 결과만)
rm cache/clustered_responses.pkl
```

### 캐시 크기 확인
```bash
du -sh cache/
```

## ⚠️ 주의사항

1. **데이터가 바뀌면 캐시 초기화**
   - 새로운 대화 추가
   - 샘플 크기 변경 (`--sample-size`)
   - → `--clear-cache` 사용

2. **모델 변경 시 캐시 초기화**
   - `model_name`을 변경한 경우
   - 임베딩 차원이 달라지므로 반드시 초기화

3. **디스크 공간 확인**
   - 캐시 파일들은 수십~수백 MB
   - 충분한 디스크 공간 필요

4. **Windows 환경**
   - 반드시 `python -X utf8` 사용 (한글 인코딩 문제)

## 🔧 프로그래밍 방식 사용

코드에서 직접 사용하려면:

```python
from analyze.cache_manager import CacheManager

# 캐시 관리자 초기화
cache = CacheManager(cache_dir="cache")

# 저장
cache.save_embeddings(response_embeddings)
cache.save_clustered_responses(clustered_responses, topic_keywords)
cache.save_document_embeddings(document_embeddings)
cache.save_similarities(similarities)

# 로드
response_embeddings = cache.load_embeddings()
clustered_responses, topic_keywords = cache.load_clustered_responses()
document_embeddings = cache.load_document_embeddings()
similarities = cache.load_similarities()

# 캐시 초기화
cache.clear_cache()

# 캐시 파일 목록
cache.list_cache_files()
```

## 📈 권장 워크플로우

```
1. 처음 실행 (캐시 생성)
   python -X utf8 test/test_embedding_workflow.py --use-cache

2. 파라미터 조정 후 재실행 (캐시 재사용)
   python -X utf8 test/test_embedding_workflow.py --use-cache

3. 데이터 변경 시 (캐시 초기화)
   python -X utf8 test/test_embedding_workflow.py --use-cache --clear-cache

4. 최종 전체 데이터 실행
   python -X utf8 test/test_embedding_workflow.py --use-cache --sample-size 376
```
