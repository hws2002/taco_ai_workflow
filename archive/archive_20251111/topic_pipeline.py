import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def load_responses(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_embeddings_pkl(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_doc_index(responses: List[Dict[str, Any]]) -> Dict[Any, int]:
    return {r["response_id"]: i for i, r in enumerate(responses)}


def align_embeddings(responses: List[Dict[str, Any]], payload: Dict[str, Any]) -> np.ndarray:
    idx = build_doc_index(responses)
    embs = np.zeros((len(responses), len(payload["items"][0]["embedding"])), dtype=np.float32)
    count = 0
    for it in payload["items"]:
        rid = it["response_id"]
        if rid in idx:
            embs[idx[rid]] = np.asarray(it["embedding"], dtype=np.float32)
            count += 1
    if count != len(responses):
        pass
    return embs


def run_bertopic(
    documents: List[str],
    embeddings: np.ndarray,
    min_topic_size: int = 15,
    top_n_words: int = 10,
    ngram_min: int = 1,
    ngram_max: int = 2,
    stop_words: str = 'english',
) -> Tuple[BERTopic, List[int]]:
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(ngram_min, ngram_max))
    model = BERTopic(
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
        vectorizer_model=vectorizer,
    )
    topics, _ = model.fit_transform(documents, embeddings)
    return model, topics


def pool_by_conversation(responses: List[Dict[str, Any]], topics: List[int], embeddings: np.ndarray) -> Dict[Any, List[Dict[str, Any]]]:
    conv_map: Dict[Any, Dict[int, List[int]]] = {}
    for i, r in enumerate(responses):
        cid = r.get("conversation_id")
        tid = topics[i]
        if tid == -1:
            continue
        conv_map.setdefault(cid, {}).setdefault(tid, []).append(i)
    conv_topic_vectors: Dict[Any, List[Dict[str, Any]]] = {}
    for cid, topic_idx in conv_map.items():
        items = []
        for tid, idxs in topic_idx.items():
            vec = embeddings[idxs].mean(axis=0)
            items.append({"topic_id": int(tid), "vector": vec, "response_ids": [responses[j]["response_id"] for j in idxs]})
        conv_topic_vectors[cid] = items
    return conv_topic_vectors


def compute_conversation_similarity(
    conv_vectors: Dict[Any, List[Dict[str, Any]]],
    hard_threshold: float = 0.8,
    pending_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    cids = list(conv_vectors.keys())
    edges = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a = cids[i]
            b = cids[j]
            va = conv_vectors.get(a, [])
            vb = conv_vectors.get(b, [])
            if not va or not vb:
                continue
            A = np.stack([x["vector"] for x in va])
            B = np.stack([x["vector"] for x in vb])
            sims = cosine_similarity(A, B)
            score = float(sims.mean())
            status = "hard" if score >= hard_threshold else ("pending" if score >= pending_threshold else None)
            if status:
                edges.append({"source": a, "target": b, "similarity": score, "status": status})
    return edges


def export_topic_info(model: BERTopic, output_dir: Path):
    topics_info = model.get_topic_info().to_dict(orient="records")
    topic_words = {}
    for t in topics_info:
        tid = int(t["Topic"]) if isinstance(t["Topic"], (int, np.integer)) else t["Topic"]
        if tid == -1:
            continue
        words = model.get_topic(tid)
        topic_words[tid] = [{"word": w, "score": float(s)} for w, s in words]
    with open(output_dir / "s3_topics.json", "w", encoding="utf-8") as f:
        json.dump({"topics": topic_words, "info": topics_info}, f, ensure_ascii=False, indent=2)


def export_doc_topics(responses: List[Dict[str, Any]], topics: List[int], output_dir: Path):
    items = []
    for r, t in zip(responses, topics):
        items.append({"response_id": r["response_id"], "conversation_id": r.get("conversation_id"), "topic_id": int(t)})
    with open(output_dir / "s3_doc_topics.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def export_conv_vectors(conv_vectors: Dict[Any, List[Dict[str, Any]]], output_dir: Path):
    out = {}
    for cid, items in conv_vectors.items():
        out[cid] = [{"topic_id": x["topic_id"], "vector": np.asarray(x["vector"]).tolist(), "response_ids": x["response_ids"]} for x in items]
    with open(output_dir / "s4_conversation_topic_vectors.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def export_edges(edges: List[Dict[str, Any]], output_dir: Path):
    hard = [e for e in edges if e["status"] == "hard"]
    pending = [e for e in edges if e["status"] == "pending"]
    with open(output_dir / "s5_edges_hard.json", "w", encoding="utf-8") as f:
        json.dump(hard, f, ensure_ascii=False, indent=2)
    with open(output_dir / "s5_edges_pending.json", "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


def export_graph(responses: List[Dict[str, Any]], edges: List[Dict[str, Any]], categories: Dict[Any, str], output_dir: Path):
    nodes = []
    for r in responses:
        nodes.append({
            "id": r["conversation_id"],
            "title": r.get("conversation_title"),
            "category": categories.get(r["conversation_id"]) if categories else None,
        })
    with open(output_dir / "s6_graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)


def load_categories(path: str) -> Dict[Any, str]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Topic pipeline: BERTopic clustering, pooling, edges, optional categories, and graph export")
    parser.add_argument("--responses", type=str, required=True, help="responses.json 경로")
    parser.add_argument("--embeddings", type=str, required=True, help="임베딩 PKL 경로")
    parser.add_argument("--output-dir", type=str, default="test/output", help="출력 디렉토리")
    parser.add_argument("--min-topic-size", type=int, default=15)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--ngram-min", type=int, default=1, help="CountVectorizer ngram 최소")
    parser.add_argument("--ngram-max", type=int, default=2, help="CountVectorizer ngram 최대")
    parser.add_argument("--stop-words", type=str, default="english", help="불용어 설정 (None, english 등)")
    parser.add_argument("--categories", type=str, default=None, help="선택: 대분류 카테고리 JSON (conversation_id -> category)")
    parser.add_argument("--hard-threshold", type=float, default=0.8, help="하드 엣지 임계값 (코사인 유사도)")
    parser.add_argument("--pending-threshold", type=float, default=0.5, help="펜딩 엣지 임계값 (코사인 유사도)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("로딩 중...")
    responses = load_responses(args.responses)
    emb_payload = load_embeddings_pkl(args.embeddings)
    X = align_embeddings(responses, emb_payload)
    docs = [r.get("content", "") for r in responses]

    print("BERTopic 클러스터링...")
    t0 = time.time()
    model, topics = run_bertopic(
        docs,
        X,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        stop_words=(None if args.stop_words.lower() == 'none' else args.stop_words),
    )
    t_bertopic = time.time() - t0
    print(f"BERTopic 완료: {t_bertopic:.2f}초")

    print("토픽 정보/할당 저장...")
    export_topic_info(model, out_dir)
    export_doc_topics(responses, topics, out_dir)

    print("대화별 토픽 벡터 풀링...")
    conv_vectors = pool_by_conversation(responses, topics, X)
    export_conv_vectors(conv_vectors, out_dir)

    print("대화간 유사도 및 엣지 생성...")
    edges = compute_conversation_similarity(
        conv_vectors,
        hard_threshold=args.hard_threshold,
        pending_threshold=args.pending_threshold,
    )
    export_edges(edges, out_dir)

    cats = load_categories(args.categories)
    print("그래프 저장...")
    export_graph(responses, edges, cats, out_dir)

    total_edges = len(edges)
    within = 0
    cross = 0
    hard_within = 0
    hard_cross = 0
    pending_within = 0
    pending_cross = 0
    if cats:
        cat_map = cats
        for e in edges:
            a = e["source"]
            b = e["target"]
            if cat_map.get(a) and cat_map.get(b):
                if cat_map[a] == cat_map[b]:
                    within += 1
                    if e.get("status") == "hard":
                        hard_within += 1
                    elif e.get("status") == "pending":
                        pending_within += 1
                else:
                    cross += 1
                    if e.get("status") == "hard":
                        hard_cross += 1
                    elif e.get("status") == "pending":
                        pending_cross += 1

    stats = {
        "bertopic_seconds": round(t_bertopic, 2),
        "total_edges": total_edges,
        "within_category_edges": within,
        "cross_category_edges": cross,
        "hard_within_category_edges": hard_within,
        "hard_cross_category_edges": hard_cross,
        "pending_within_category_edges": pending_within,
        "pending_cross_category_edges": pending_cross,
    }
    with open(out_dir / "s6_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("완료")


if __name__ == "__main__":
    main()
