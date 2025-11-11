"""
LLM 기반 대분류 클러스터링
키워드를 기반으로 LLM이 의미론적 클러스터를 생성
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv


class LLMClusterer:
    """
    LLM을 사용한 키워드 기반 대분류 클러스터링

    지원하는 LLM:
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude)
    """

    def __init__(
        self,
        provider: str = "openai",  # "openai" or "anthropic"
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Args:
            provider: LLM 제공자 ("openai" 또는 "anthropic")
            model: 사용할 모델명
            api_key: API 키 (None이면 환경변수에서 가져옴)
            prompt_template: 커스텀 프롬프트 템플릿
        """
        self.provider = provider.lower()
        self.model = model

        # API 키 설정
        current_file_path = Path(__file__)
        script_dir = current_file_path.resolve().parent
        dotenv_path = script_dir / '.env'
        load_dotenv(dotenv_path=dotenv_path)

        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}")
        
        print(self.api_key)

        if not self.api_key:
            raise ValueError(f"{self.provider.upper()} API 키가 설정되지 않았습니다.")

        # 프롬프트 템플릿
        self.prompt_template = prompt_template or self._get_default_prompt()

        # API 클라이언트 초기화
        self._init_client()

    def _init_client(self):
        """API 클라이언트 초기화"""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    baseURL="https://api.chatanywhere.tech/v1"
                )
            except ImportError:
                raise ImportError("OpenAI 패키지가 설치되어 있지 않습니다: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic 패키지가 설치되어 있지 않습니다: pip install anthropic")

    def _get_default_prompt(self) -> str:
        """기본 프롬프트 템플릿 반환"""
        return """당신은 대화 데이터를 분석하는 전문가입니다.

각 채팅 세션의 주요 키워드 목록이 제공됩니다. 이 키워드들을 분석하여 **3~5개의 의미 있는 대분류 클러스터**를 만들고, 각 채팅이 어느 클러스터에 속하는지 판단해주세요.

**입력 데이터:**
{input_data}

**요구사항:**
1. 3~5개의 대분류 클러스터를 생성하세요
2. 각 클러스터는 명확한 주제나 카테고리를 나타내야 합니다
3. cluster_id는 cluster_A, cluster_B, cluster_C 형식을 사용하세요
4. 각 클러스터 설명은 "한글 이름 (English Name)" 형식으로 작성하세요
5. 각 채팅을 가장 적합한 클러스터에 할당하세요
6. 모든 채팅은 정확히 하나의 클러스터에만 할당되어야 합니다

**출력 형식 (반드시 JSON만 출력):**
```json
{
  "cluster_definitions": {
    "cluster_A": "날씨 및 기상 정보 (Weather & Forecast)",
    "cluster_B": "요리 및 레시피 (Cooking & Recipes)",
    "cluster_C": "스포츠 (Sports)"
  },
  "chat_assignments": [
    {
      "chat_id": "session_001",
      "cluster_id": "cluster_A"
    },
    {
      "chat_id": "session_002",
      "cluster_id": "cluster_B"
    }
  ]
}
```

**주의:**
- 반드시 위 JSON 형식으로만 응답하세요
- 추가 설명이나 마크다운 없이 JSON만 출력
- cluster_definitions와 chat_assignments 두 필드만 포함
- 모든 채팅은 정확히 하나의 클러스터에만 할당
"""

    def cluster_chats(
        self,
        chats_data: Dict[str, Any],
        n_clusters: Optional[int] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        LLM을 사용하여 채팅 데이터를 클러스터링

        Args:
            chats_data: 채팅 키워드 데이터
                {
                    "chats": [
                        {
                            "chat_id": "session_001",
                            "keywords": [
                                {"word": "키워드", "score": 0.95}
                            ]
                        }
                    ]
                }
            n_clusters: 생성할 클러스터 개수 (None이면 LLM이 자동 결정)
            temperature: LLM 샘플링 온도 (낮을수록 일관적)

        Returns:
            클러스터링 결과
            {
                "clusters": [...],
                "chat_assignments": [...],
                "summary": {...}
            }
        """
        print(f"\n{'='*80}")
        print(f"LLM 기반 대분류 클러스터링")
        print(f"{'='*80}")
        print(f"제공자: {self.provider}")
        print(f"모델: {self.model}")
        print(f"채팅 수: {len(chats_data.get('chats', []))}")

        # 입력 데이터 준비
        input_json = json.dumps(chats_data, ensure_ascii=False, indent=2)

        # 프롬프트 생성
        prompt = self.prompt_template.format(input_data=input_json)

        if n_clusters:
            prompt += f"\n\n**중요: 정확히 {n_clusters}개의 클러스터를 생성해야 합니다.**"

        # LLM 호출
        print(f"\nLLM에게 클러스터링 요청 중...")

        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, temperature)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt, temperature)
            else:
                raise ValueError(f"지원하지 않는 provider: {self.provider}")

            # JSON 파싱
            result = self._parse_response(response)

            print(f"✓ 클러스터링 완료")
            print(f"  - 생성된 클러스터: {len(result['cluster_definitions'])}개")
            print(f"  - 할당된 채팅: {len(result['chat_assignments'])}개")

            # 클러스터 정보 출력
            print(f"\n생성된 클러스터:")
            for cluster_id, description in result['cluster_definitions'].items():
                print(f"\n  [{cluster_id}] {description}")

            # 분포 계산 및 출력
            from collections import Counter
            cluster_counts = Counter(assignment['cluster_id']
                                    for assignment in result['chat_assignments'])

            print(f"\n클러스터 분포:")
            for cluster_id, count in sorted(cluster_counts.items()):
                print(f"  {cluster_id}: {count}개")

            return result

        except Exception as e:
            print(f"❌ 클러스터링 실패: {e}")
            raise

    def _call_openai(self, prompt: str, temperature: float) -> str:
        """OpenAI API 호출"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 데이터 분석 전문가입니다. 항상 정확한 JSON 형식으로만 응답합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            response_format={"type": "json_object"}  # JSON 모드 강제
        )

        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        """Anthropic API 호출"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답을 JSON으로 파싱"""
        # JSON 코드 블록 제거 (```json ... ``` 형식)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        # JSON 파싱
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패. 원본 응답:\n{response}")
            raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {e}")

        # 필수 필드 검증
        if "cluster_definitions" not in result:
            raise ValueError("응답에 'cluster_definitions' 필드가 없습니다")
        if "chat_assignments" not in result:
            raise ValueError("응답에 'chat_assignments' 필드가 없습니다")

        return result

    def save_result(self, result: Dict[str, Any], output_path: str):
        """클러스터링 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 결과 저장: {output_path}")


def extract_keywords_from_embeddings(
    document_embeddings: Dict[int, Any],
    response_embeddings: Dict[str, Dict[str, Any]],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    문서 임베딩에서 키워드 추출

    Args:
        document_embeddings: {conv_id: DocumentEmbedding}
        response_embeddings: {response_id: {concepts, ...}}
        top_k: 각 대화당 상위 K개 키워드

    Returns:
        LLM 입력 형식의 데이터
    """
    chats_data = {"chats": []}

    for conv_id, doc_emb in document_embeddings.items():
        # 해당 대화의 모든 키워드 수집
        all_keywords = []

        # response_embeddings에서 해당 대화의 키워드 추출
        for resp_id, resp_data in response_embeddings.items():
            if resp_data.get('conversation_id') == conv_id:
                concepts = resp_data.get('concepts', [])
                all_keywords.extend(concepts)

        # 빈도수 계산
        from collections import Counter
        keyword_counts = Counter(all_keywords)

        # 상위 K개 선택 (점수는 빈도 기반으로 정규화)
        total_count = sum(keyword_counts.values())
        top_keywords = [
            {
                "word": word,
                "score": round(count / total_count, 3)
            }
            for word, count in keyword_counts.most_common(top_k)
        ]

        chats_data["chats"].append({
            "chat_id": f"conv_{conv_id}",
            "title": doc_emb.conversation_title,
            "keywords": top_keywords
        })

    return chats_data


if __name__ == "__main__":
    # 예제 데이터
    example_data = {
        "chats": [
            {
                "chat_id": "session_001",
                "keywords": [
                    {"word": "미세먼지", "score": 0.952},
                    {"word": "날씨", "score": 0.923},
                    {"word": "주말", "score": 0.891},
                    {"word": "서울", "score": 0.885},
                    {"word": "비", "score": 0.810}
                ]
            },
            {
                "chat_id": "session_002",
                "keywords": [
                    {"word": "파스타", "score": 0.971},
                    {"word": "레시피", "score": 0.945},
                    {"word": "알리오 올리오", "score": 0.912},
                    {"word": "재료", "score": 0.850},
                    {"word": "요리", "score": 0.833}
                ]
            },
            {
                "chat_id": "session_003",
                "keywords": [
                    {"word": "손흥민", "score": 0.988},
                    {"word": "축구", "score": 0.960},
                    {"word": "이강인", "score": 0.931},
                    {"word": "프리미어리그", "score": 0.905}
                ]
            }
        ]
    }

    # 테스트
    try:
        clusterer = LLMClusterer(
            provider="openai",
            model="gpt-4"
        )

        result = clusterer.cluster_chats(example_data)

        # 결과 저장
        clusterer.save_result(result, "output/llm_clustering_result.json")

        print("\n✅ 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
