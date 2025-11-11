@echo off
echo ============================================
echo 키워드 추출 테스트 스크립트
echo ============================================
echo.

echo [1/3] 필요한 라이브러리 설치 중...
pip install jieba langdetect
echo.

echo [2/3] 테스트 모드로 키워드 추출 실행 중...
echo (각 언어별 50개 conversation 샘플링)
echo.

python extract_keywords_fast.py ^
    --input test/output_full/s1_ai_responses.json ^
    --output test/output_full/s2_ai_responses_with_keywords_test.json ^
    --embeddings test/output_full/s2_conversation_embeddings_paraphrase-multilingual-mpnet-base-v2.pkl ^
    --test-mode ^
    --samples-per-language 50 ^
    --top-n 5 ^
    --diversity 0.7 ^
    --nr-candidates 20 ^
    --processes 4

echo.
echo [3/3] 완료!
echo.
echo 결과 파일: test/output_full/s2_ai_responses_with_keywords_test.json
echo.
pause
