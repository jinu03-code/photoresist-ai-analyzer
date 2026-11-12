# Pattern CD Analyzer

포토레지스트 원형 패턴의 CD 측정, 조건별 비교, 데이터 기반 자동 총평 및 공정변수 조절 방향을 제공하는 Streamlit 앱입니다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 주요 기능
- 이미지 업로드 및 CD 측정
- 목표 CD/허용오차 기반 판정
- 조건별 평균 CD, 수율, 편차 비교
- 규칙 기반 자동 총평 (API 키 및 유료 AI 불필요)
- Exposure, Spin coating, Soft bake, IRB, Development, DI rinse의 후속 실험 조절 방향 제안
- 결과 CSV 및 분석 이미지 ZIP 다운로드

## 공정변수 제안 해석 시 주의
자동 총평은 측정 데이터와 사전 정의된 규칙에 따른 실험 설계 보조 기능입니다. 제시하는 조절 폭은 보편적 최적값이 아니며, 레지스트 종류/톤, 마스크 패턴, 장비 및 제조사 공정 사양에 따라 달라집니다. 한 번에 한 변수만 변경하고 반복 측정으로 검증하세요. 광학 2D 이미지로 막 두께, 측벽, 언더컷을 직접 판정하지 않습니다.

## 데이터 보존
분석 결과는 실행 폴더의 `saved_data`에 저장됩니다. 코드 교체 시 기존 `saved_data` 폴더는 삭제하지 마세요.
