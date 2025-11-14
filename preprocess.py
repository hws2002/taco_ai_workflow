import re
import emoji # (설치 필요: pip install emoji)
import jieba # (설치 필요: pip install jieba)

# [핵심 1] re.sub의 콜백 함수 정의
# 텍스트 스캔 중 중국어 블록을 찾으면(match) 이 함수가 호출됩니다.
def _apply_jieba_to_chinese(match):
    """
    re.sub가 찾은 중국어 텍스트 블록(match.group(0))을 받아서
    jieba로 띄어쓰기 처리한 후 반환합니다.
    """
    chinese_text = match.group(0)
    tokenized_text = " ".join(jieba.cut(chinese_text))
    return tokenized_text

def preprocess_content(text):
    """
    NLP 모델(KeyBERT, SBERT 등) 입력을 위해 텍스트에서 노이즈를 제거합니다.
    
    1. 코드 블록 (```...```) 제거
    2. URL (http..., www...) 제거
    3. 이모티콘 (😂, ️ 등) 제거
    4. [수정] 광범위한 문장 부호 및 특수기호 제거
    5. 인용/출처 태그 (예: 【7†source】) 제거
    6. 마크다운 수평선 (---) 제거
    7. [핵심] "중국어" 블록에만 선택적으로 jieba 띄어쓰기 적용
    8. 모든 공백(줄바꿈, 탭, 여러 공백)을 단일 공백으로 변환
    """
    
    # 1. 코드 블록 제거 (언어 태그 포함, 3개 이상 백틱/틸드, 동일 길이 펜스 매칭)
    # ```python ... ``` 또는 ```` ... ```` 등
    text = re.sub(r'(?s)(?m)^(?P<fence>`{3,})[^\n]*\n.*?^\1\s*$', '', text)
    text = re.sub(r'(?s)(?m)^(?P<fence>~{3,})[^\n]*\n.*?^\1\s*$', '', text)
    text = re.sub(r'<pre[\s\S]*?>[\s\S]*?</pre>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<code[\s\S]*?>[\s\S]*?</code>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'(?m)^(?: {4}|\t).*$',' ', text)
    
    # 2. URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. 이모티콘 제거
    text = emoji.replace_emoji(text, replace='')
    
    # 4. [✨ 수정] 마크다운, 특수기호 및 각종 문장 부호 제거
    #    (이전 요청의 』『。() 등 포함)
    PUNCT_TO_REMOVE = r'[\*\-→·/》，、：《』『。，“”"\'（）()?\[\]{}!.;:%]'
    text = re.sub(PUNCT_TO_REMOVE, '', text)
    
    # 5. 인용/출처 태그 (【...】) 제거
    text = re.sub(r'【.*?】', '', text)
    
    # 6. 마크다운 수평선 (---) 제거
    text = re.sub(r'--+', '', text) # 하이픈 2개 이상 연속된 것
    
    # 7. [✨ 핵심 2] 중국어(CJK) 유니코드 블록에만 _apply_jieba_to_chinese 함수 적용
    #    r'[\u4e00-\u9fff]+' : CJK 통합 한자 영역의 연속된 블록을 찾습니다.
    #    (기존의 7번 라인을 이것으로 대체합니다)
    chinese_char_pattern = r'[\u4e00-\u9fff]+'
    text = re.sub(chinese_char_pattern, _apply_jieba_to_chinese, text)
    
    # 8. 모든 공백 문자(\n, \t, 여러 공백)를 단일 공백(' ')으로 치환
    text = re.sub(r'\s+', ' ', text)
    
    # 9. 양쪽 끝에 남은 공백 제거
    return text.strip()

# --- 실행 예시 ---

# 사용자님이 제공한 JSON 데이터 샘플
json_data = [
 {
  "response_id": "288_1",
  "conversation_id": 288,
  "content": "该文档名为《国际关系理论研究的困境、进展与前景》，作者为刘丰。Despite these challenges, the paper notes that significant efforts are still being made in the field. Researchers are exploring local knowledge, developing mid-level theories, seeking theoretical integration and dialogue, and drawing on interdisciplinary approaches. 文中讨论了国际关系理论领域的当前状态和挑战。摘要部分指出，国际关系学界普遍存在对主义的怀疑、理论发展的终结以及学科衰落的论调。국제관계 이론을 연구하는 과정에서 학자들은 종종 이론의 실용성과 설명력에 대한 의문을 마주한다. 이러한 현상은 류펑(刘丰)의 『국제관계 이론 연구의 난관, 진전 및 전망』을 읽기 전에는 특히 두드러졌다. 류펑은 현재 국제관계 이론이 처한 난관을 심도 있게 분석했으며, 특히 학계가 이론 혁신에 대해 보이는 비관적 태도를 비판함으로써 기존 관점을 도전했다. 이론 혁신의 동력, 주기 및 패턴을 깊이 있게 분석함으로써 류펑은 이론 연구가 여전히 중요한 가치를 지닌다는 점을 밝혀냈다. 这种看法源于对理论创新动力、周期和模式的认识偏差，以及受到大辩论学科史叙事方式的误导。\n\n尽管面临这些挑战，但仍有研究人员在积极探索，例如研究地方性知识、发展中层理论、寻求理论综合与对话以及跨学科借鉴。这些努力对未来国际关系理论的发展和进步至"
 }
]

# --- 전처리 실행 ---
print("--- 전처리 결과 미리보기 ---")
for item in json_data:
    original_content = item['content']
    cleaned_content = preprocess_content(original_content)
    
    print(f"\n--- ID: {item['response_id']} 원본 ---")
    print(original_content)
    print(f"\n--- ID: {item['response_id']} 전처리 후 (최종본) ---")
    print(cleaned_content)
    print("="*30)