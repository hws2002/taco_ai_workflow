@echo off
echo ============================================
echo 통합 키워드 추출 파이프라인 (전체 데이터)
echo ============================================
echo.

echo Conda 환경 활성화: taco
call conda activate taco
echo.

echo ============================================
echo 전체 데이터 실행
echo ============================================
echo.

python pipeline_extract_keywords.py ^
    --input test/output_full/s1_ai_responses.json ^
    --output test/output_full/s2_ai_responses_with_keywords.json ^
    --save-embeddings test/output_full/conversation_embeddings.pkl ^
    --embedding-model thenlper/gte-base ^
    --keyword-model thenlper/gte-base ^
    --cache-dir models_cache ^
    --top-n 5

echo.
echo ============================================
echo 결과 확인
echo ============================================
echo.

python check_keywords_result.py ^
    --file test/output_full/s2_ai_responses_with_keywords.json ^
    --samples 5

echo.
echo 완료!
echo.
pause
