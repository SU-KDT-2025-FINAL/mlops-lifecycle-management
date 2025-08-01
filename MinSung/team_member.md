## 👥 5. 역할 분담 (2명 백엔드 + 2명 데이터 엔지니어)

| 팀원 | 역할 | 상세 작업 |
|------|------|-----------|
| 👨‍💻 백엔드 A | API / 유저 인터페이스 연결 | - 고객 요청 수신 API<br>- 유저 ID 전달 → 모델 예측 → 결과 반환<br>- CRM 액션 트리거 API 호출 (쿠폰 발송 등)<br>- Streamlit / FastAPI UI 연동 |
| 👨‍💻 백엔드 B | MLOps 배포 자동화 | - 모델 inference 서버 구성<br>- 모델 버전관리 (SageMaker or Flask)<br>- Kafka → 모델 추론 자동화 연동<br>- 로깅 / 에러 처리 시스템 구성 |
| 🧑‍🔬 데이터 엔지니어 A | 데이터 전처리 + Feature 설계 | - 고객 로그 → RFM + 행동 feature 설계<br>- 스케일링 / 인코딩 / 이상값 처리<br>- 데이터 배치 파이프라인 (Spark, Pandas) 구축<br>- Lake Formation + Glue 작업 |
| 🧑‍🔬 데이터 엔지니어 B | 모델링 + 평가 + 대시보드 | - 고객 이탈 예측 모델 학습 (XGBoost, LGBM)<br>- 클러스터링 모델 (KMeans) 학습 및 해석<br>- 정책 분류 로직 설계 (군집 + 이탈율 기반)<br>- 대시보드 시각화 (Plotly, Tableau 등) |
