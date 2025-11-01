"""
캐시 관리 모듈
임베딩과 BERTopic 결과를 저장하고 불러오는 기능
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime


class CacheManager:
    """캐시 저장 및 로드 관리 클래스"""

    def __init__(self, cache_dir: str = "cache"):
        """
        Args:
            cache_dir: 캐시 파일을 저장할 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
        self,
        response_embeddings: Dict[str, Dict[str, Any]],
        filename: str = "response_embeddings.pkl"
    ):
        """
        AI 답변 임베딩을 저장

        Args:
            response_embeddings: {response_id: {embedding, concepts, ...}}
            filename: 저장할 파일명
        """
        filepath = self.cache_dir / filename

        print(f"\n임베딩 캐시 저장 중: {filepath}")

        # 메타데이터 추가
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'count': len(response_embeddings),
            'embeddings': response_embeddings
        }

        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✓ {len(response_embeddings)}개의 임베딩 저장 완료")
        print(f"  파일 크기: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    def load_embeddings(
        self,
        filename: str = "response_embeddings.pkl"
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        저장된 임베딩을 로드

        Args:
            filename: 로드할 파일명

        Returns:
            response_embeddings 또는 None (파일이 없으면)
        """
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"캐시 파일이 없습니다: {filepath}")
            return None

        print(f"\n임베딩 캐시 로드 중: {filepath}")

        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)

        embeddings = cache_data['embeddings']
        timestamp = cache_data.get('timestamp', 'Unknown')

        print(f"✓ {len(embeddings)}개의 임베딩 로드 완료")
        print(f"  생성 시간: {timestamp}")

        return embeddings

    def save_bertopic_model(
        self,
        topic_model,
        filename: str = "bertopic_model"
    ):
        """
        BERTopic 모델 저장

        Args:
            topic_model: BERTopic 모델
            filename: 저장할 파일명 (확장자 없이)
        """
        filepath = self.cache_dir / filename

        print(f"\nBERTopic 모델 저장 중: {filepath}")

        try:
            topic_model.save(str(filepath), serialization="pickle")
            print(f"✓ BERTopic 모델 저장 완료")
        except Exception as e:
            print(f"경고: BERTopic 모델 저장 실패 - {e}")

    def load_bertopic_model(
        self,
        filename: str = "bertopic_model"
    ):
        """
        저장된 BERTopic 모델 로드

        Args:
            filename: 로드할 파일명 (확장자 없이)

        Returns:
            BERTopic 모델 또는 None
        """
        from bertopic import BERTopic

        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"BERTopic 모델 캐시 파일이 없습니다: {filepath}")
            return None

        print(f"\nBERTopic 모델 로드 중: {filepath}")

        try:
            topic_model = BERTopic.load(str(filepath))
            print(f"✓ BERTopic 모델 로드 완료")
            return topic_model
        except Exception as e:
            print(f"경고: BERTopic 모델 로드 실패 - {e}")
            return None

    def save_clustered_responses(
        self,
        clustered_responses: Dict,
        topic_keywords: Dict,
        filename: str = "clustered_responses.pkl"
    ):
        """
        클러스터링 결과 저장

        Args:
            clustered_responses: {response_id: ClusteredResponse}
            topic_keywords: {topic_id: [keywords]}
            filename: 저장할 파일명
        """
        filepath = self.cache_dir / filename

        print(f"\n클러스터링 결과 저장 중: {filepath}")

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'clustered_responses': clustered_responses,
            'topic_keywords': topic_keywords
        }

        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✓ 클러스터링 결과 저장 완료")

    def load_clustered_responses(
        self,
        filename: str = "clustered_responses.pkl"
    ) -> Optional[Tuple[Dict, Dict]]:
        """
        클러스터링 결과 로드

        Args:
            filename: 로드할 파일명

        Returns:
            (clustered_responses, topic_keywords) 또는 None
        """
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"클러스터링 캐시 파일이 없습니다: {filepath}")
            return None

        print(f"\n클러스터링 결과 로드 중: {filepath}")

        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)

        clustered_responses = cache_data['clustered_responses']
        topic_keywords = cache_data['topic_keywords']
        timestamp = cache_data.get('timestamp', 'Unknown')

        print(f"✓ 클러스터링 결과 로드 완료")
        print(f"  생성 시간: {timestamp}")

        return clustered_responses, topic_keywords

    def save_document_embeddings(
        self,
        document_embeddings: Dict,
        filename: str = "document_embeddings.pkl"
    ):
        """
        문서별 풀링된 임베딩 저장

        Args:
            document_embeddings: {conversation_id: DocumentEmbedding}
            filename: 저장할 파일명
        """
        filepath = self.cache_dir / filename

        print(f"\n문서 임베딩 저장 중: {filepath}")

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'document_embeddings': document_embeddings
        }

        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✓ {len(document_embeddings)}개의 문서 임베딩 저장 완료")

    def load_document_embeddings(
        self,
        filename: str = "document_embeddings.pkl"
    ) -> Optional[Dict]:
        """
        문서별 풀링된 임베딩 로드

        Args:
            filename: 로드할 파일명

        Returns:
            document_embeddings 또는 None
        """
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"문서 임베딩 캐시 파일이 없습니다: {filepath}")
            return None

        print(f"\n문서 임베딩 로드 중: {filepath}")

        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)

        document_embeddings = cache_data['document_embeddings']
        timestamp = cache_data.get('timestamp', 'Unknown')

        print(f"✓ {len(document_embeddings)}개의 문서 임베딩 로드 완료")
        print(f"  생성 시간: {timestamp}")

        return document_embeddings

    def save_similarities(
        self,
        similarities: Dict,
        filename: str = "similarities.pkl"
    ):
        """
        유사도 결과 저장

        Args:
            similarities: {(conv_id_1, conv_id_2): similarity}
            filename: 저장할 파일명
        """
        filepath = self.cache_dir / filename

        print(f"\n유사도 결과 저장 중: {filepath}")

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'similarities': similarities
        }

        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✓ {len(similarities)}개의 유사도 쌍 저장 완료")

    def load_similarities(
        self,
        filename: str = "similarities.pkl"
    ) -> Optional[Dict]:
        """
        유사도 결과 로드

        Args:
            filename: 로드할 파일명

        Returns:
            similarities 또는 None
        """
        filepath = self.cache_dir / filename

        if not filepath.exists():
            print(f"유사도 캐시 파일이 없습니다: {filepath}")
            return None

        print(f"\n유사도 결과 로드 중: {filepath}")

        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)

        similarities = cache_data['similarities']
        timestamp = cache_data.get('timestamp', 'Unknown')

        print(f"✓ {len(similarities)}개의 유사도 쌍 로드 완료")
        print(f"  생성 시간: {timestamp}")

        return similarities

    def clear_cache(self):
        """캐시 디렉토리의 모든 파일 삭제"""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 캐시 디렉토리 초기화: {self.cache_dir}")

    def list_cache_files(self):
        """캐시 디렉토리의 파일 목록 출력"""
        print(f"\n캐시 디렉토리: {self.cache_dir}")
        print("-" * 80)

        if not self.cache_dir.exists():
            print("캐시 디렉토리가 없습니다.")
            return

        files = list(self.cache_dir.glob("*"))

        if not files:
            print("캐시 파일이 없습니다.")
            return

        total_size = 0
        for file in sorted(files):
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += size_mb
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  {file.name:40s} {size_mb:8.2f} MB  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        print("-" * 80)
        print(f"총 {len(files)}개 파일, {total_size:.2f} MB")
