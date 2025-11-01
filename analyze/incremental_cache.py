"""
증분 캐싱 모듈 (Incremental Caching)
메시지 ID 기반으로 임베딩과 유사도를 저장/로드
새로운 메시지 추가 시 기존 캐시 재사용
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List
import numpy as np
from datetime import datetime


class IncrementalCache:
    """
    ID 기반 증분 캐싱 클래스

    특징:
    - 메시지 ID를 키로 사용
    - 새 데이터 추가 시 기존 캐시 재사용
    - 부분 업데이트 가능
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Args:
            cache_dir: 캐시 파일을 저장할 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 캐시 파일 경로
        self.embeddings_file = self.cache_dir / "embeddings_by_id.pkl"
        self.similarities_file = self.cache_dir / "similarities_by_id.pkl"
        self.clustering_file = self.cache_dir / "clustering_by_id.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"

    # ============================================================
    # 임베딩 캐시
    # ============================================================

    def load_embeddings_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        임베딩 캐시 로드

        Returns:
            {response_id: {embedding, concepts, conversation_id, ...}}
        """
        if not self.embeddings_file.exists():
            return {}

        with open(self.embeddings_file, 'rb') as f:
            cache = pickle.load(f)

        print(f"✓ 임베딩 캐시 로드: {len(cache)}개")
        return cache

    def save_embeddings_cache(
        self,
        embeddings_cache: Dict[str, Dict[str, Any]]
    ):
        """
        임베딩 캐시 저장

        Args:
            embeddings_cache: {response_id: {embedding, concepts, ...}}
        """
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_cache, f)

        print(f"✓ 임베딩 캐시 저장: {len(embeddings_cache)}개")

    def get_missing_embeddings(
        self,
        all_response_ids: List[str],
        embeddings_cache: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        캐시에 없는 응답 ID 목록 반환

        Args:
            all_response_ids: 필요한 모든 응답 ID
            embeddings_cache: 현재 캐시

        Returns:
            캐시에 없는 응답 ID 리스트
        """
        cached_ids = set(embeddings_cache.keys())
        all_ids = set(all_response_ids)
        missing_ids = all_ids - cached_ids

        return list(missing_ids)

    def update_embeddings_cache(
        self,
        embeddings_cache: Dict[str, Dict[str, Any]],
        new_embeddings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        캐시에 새로운 임베딩 추가

        Args:
            embeddings_cache: 기존 캐시
            new_embeddings: 새로 계산한 임베딩

        Returns:
            업데이트된 캐시
        """
        embeddings_cache.update(new_embeddings)
        return embeddings_cache

    # ============================================================
    # 유사도 캐시
    # ============================================================

    def load_similarities_cache(self) -> Dict[Tuple[int, int], float]:
        """
        유사도 캐시 로드

        Returns:
            {(conv_id_1, conv_id_2): similarity}
        """
        if not self.similarities_file.exists():
            return {}

        with open(self.similarities_file, 'rb') as f:
            cache = pickle.load(f)

        print(f"✓ 유사도 캐시 로드: {len(cache)}개 쌍")
        return cache

    def save_similarities_cache(
        self,
        similarities_cache: Dict[Tuple[int, int], float]
    ):
        """
        유사도 캐시 저장

        Args:
            similarities_cache: {(conv_id_1, conv_id_2): similarity}
        """
        with open(self.similarities_file, 'wb') as f:
            pickle.dump(similarities_cache, f)

        print(f"✓ 유사도 캐시 저장: {len(similarities_cache)}개 쌍")

    def get_missing_similarities(
        self,
        all_conv_ids: List[int],
        similarities_cache: Dict[Tuple[int, int], float]
    ) -> List[Tuple[int, int]]:
        """
        캐시에 없는 유사도 쌍 목록 반환

        Args:
            all_conv_ids: 모든 대화 ID
            similarities_cache: 현재 캐시

        Returns:
            캐시에 없는 (conv_id_1, conv_id_2) 쌍 리스트
        """
        # 계산해야 할 모든 쌍
        all_pairs = set()
        for i, conv_id_1 in enumerate(all_conv_ids):
            for conv_id_2 in all_conv_ids[i + 1:]:
                pair = tuple(sorted([conv_id_1, conv_id_2]))
                all_pairs.add(pair)

        # 캐시에 있는 쌍
        cached_pairs = set()
        for (id1, id2) in similarities_cache.keys():
            pair = tuple(sorted([id1, id2]))
            cached_pairs.add(pair)

        # 캐시에 없는 쌍
        missing_pairs = all_pairs - cached_pairs

        return list(missing_pairs)

    def update_similarities_cache(
        self,
        similarities_cache: Dict[Tuple[int, int], float],
        new_similarities: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], float]:
        """
        캐시에 새로운 유사도 추가

        Args:
            similarities_cache: 기존 캐시
            new_similarities: 새로 계산한 유사도

        Returns:
            업데이트된 캐시
        """
        similarities_cache.update(new_similarities)
        return similarities_cache

    # ============================================================
    # 클러스터링 캐시
    # ============================================================

    def load_clustering_cache(self) -> Dict[str, int]:
        """
        클러스터링 결과 캐시 로드

        Returns:
            {response_id: topic_id}
        """
        if not self.clustering_file.exists():
            return {}

        with open(self.clustering_file, 'rb') as f:
            cache = pickle.load(f)

        print(f"✓ 클러스터링 캐시 로드: {len(cache)}개")
        return cache

    def save_clustering_cache(
        self,
        clustering_cache: Dict[str, int]
    ):
        """
        클러스터링 결과 캐시 저장

        Args:
            clustering_cache: {response_id: topic_id}
        """
        with open(self.clustering_file, 'wb') as f:
            pickle.dump(clustering_cache, f)

        print(f"✓ 클러스터링 캐시 저장: {len(clustering_cache)}개")

    # ============================================================
    # 메타데이터
    # ============================================================

    def save_metadata(
        self,
        metadata: Dict[str, Any]
    ):
        """
        캐시 메타데이터 저장 (통계 정보)

        Args:
            metadata: {
                'last_update': timestamp,
                'total_responses': count,
                'total_conversations': count,
                ...
            }
        """
        metadata['last_update'] = datetime.now().isoformat()

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """캐시 메타데이터 로드"""
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ============================================================
    # 통계 및 유틸리티
    # ============================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        embeddings_cache = self.load_embeddings_cache()
        similarities_cache = self.load_similarities_cache()
        clustering_cache = self.load_clustering_cache()
        metadata = self.load_metadata()

        stats = {
            'embeddings_count': len(embeddings_cache),
            'similarities_count': len(similarities_cache),
            'clustering_count': len(clustering_cache),
            'metadata': metadata
        }

        return stats

    def clear_cache(self):
        """모든 캐시 파일 삭제"""
        files_to_remove = [
            self.embeddings_file,
            self.similarities_file,
            self.clustering_file,
            self.metadata_file
        ]

        for file in files_to_remove:
            if file.exists():
                file.unlink()

        print("✓ 모든 캐시 파일 삭제 완료")

    def print_cache_info(self):
        """캐시 정보 출력"""
        print("\n" + "=" * 80)
        print("캐시 정보")
        print("=" * 80)

        stats = self.get_cache_stats()

        print(f"임베딩 캐시: {stats['embeddings_count']}개")
        print(f"유사도 캐시: {stats['similarities_count']}개 쌍")
        print(f"클러스터링 캐시: {stats['clustering_count']}개")

        if stats['metadata']:
            print(f"\n메타데이터:")
            for key, value in stats['metadata'].items():
                print(f"  {key}: {value}")

        print("=" * 80)

    # ============================================================
    # 헬퍼 함수
    # ============================================================

    def get_embedding_hit_rate(
        self,
        all_response_ids: List[str]
    ) -> float:
        """
        임베딩 캐시 적중률 계산

        Returns:
            적중률 (0~1)
        """
        embeddings_cache = self.load_embeddings_cache()

        if not all_response_ids:
            return 0.0

        cached_count = sum(1 for rid in all_response_ids if rid in embeddings_cache)
        hit_rate = cached_count / len(all_response_ids)

        return hit_rate

    def get_similarity_hit_rate(
        self,
        all_conv_ids: List[int]
    ) -> float:
        """
        유사도 캐시 적중률 계산

        Returns:
            적중률 (0~1)
        """
        similarities_cache = self.load_similarities_cache()

        if len(all_conv_ids) < 2:
            return 0.0

        # 전체 필요한 쌍 수
        total_pairs = len(all_conv_ids) * (len(all_conv_ids) - 1) // 2

        # 캐시에 있는 쌍 수
        cached_pairs = 0
        for i, conv_id_1 in enumerate(all_conv_ids):
            for conv_id_2 in all_conv_ids[i + 1:]:
                pair = tuple(sorted([conv_id_1, conv_id_2]))
                if pair in similarities_cache:
                    cached_pairs += 1

        hit_rate = cached_pairs / total_pairs if total_pairs > 0 else 0.0

        return hit_rate

    def estimate_cache_size(self) -> Dict[str, float]:
        """캐시 파일 크기 추정 (MB)"""
        sizes = {}

        files = {
            'embeddings': self.embeddings_file,
            'similarities': self.similarities_file,
            'clustering': self.clustering_file,
            'metadata': self.metadata_file
        }

        for name, file_path in files.items():
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                sizes[name] = size_mb
            else:
                sizes[name] = 0.0

        sizes['total'] = sum(sizes.values())

        return sizes
