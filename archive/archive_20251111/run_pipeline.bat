@echo off
echo ============================================
echo 통합 키워드 추출 파이프라인
echo ============================================
echo.

echo Conda 환경 활성화: taco
call conda activate taco
echo.

echo 필요한 라이브러리 설치 중...
pip install jieba langdetect
echo.

echo ============================================
echo 테스트 모드 실행 (10개 conversation)
echo ============================================
echo.

python pipeline_extract_keywords.py ^
    --input test/output_full/s1_ai_responses.json ^
    --output test/output_full/s2_keywords_pipeline_test.json ^
    --save-embeddings test/output_full/conversation_embeddings_pipeline_test.pkl ^
    --embedding-model thenlper/gte-base ^
    --keyword-model thenlper/gte-base ^
    --cache-dir models_cache ^
    --top-n 5 ^
    --test-mode

echo.
echo ============================================
echo 결과 확인
echo ============================================
echo.

python check_keywords_result.py ^
    --file test/output_full/s2_keywords_pipeline_test.json ^
    --samples 3

echo.
echo 완료!
echo.
pause
