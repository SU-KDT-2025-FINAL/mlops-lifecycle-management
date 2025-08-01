# 🚀 AI 예측 분석 SaaS를 위한 MLOps 적용 방안

본 문서는 'AI 예측 분석 SaaS' 프로젝트에 MLOps를 성공적으로 적용하기 위한 핵심 목표와 단계별 파이프라인 구축 방안을 제시합니다.

---

## 🎯 MLOps의 핵심 목표

1.  **자동화 (Automation)**: 신규 고객사 가입 시 데이터 수집부터 모델 훈련, 예측까지의 전 과정을 **최소한의 개입으로 자동 실행**합니다.
2.  **재현성 (Reproducibility)**: "어떤 데이터와 코드로 이 예측 결과가 나왔지?"를 언제든 추적하여 문제 해결과 신뢰도를 확보합니다.
3.  **확장성 (Scalability)**: 고객사가 10개에서 1000개로 늘어나도 전체 시스템이 안정적으로 운영될 수 있는 구조를 만듭니다.
4.  **모니터링 (Monitoring)**: 배포된 모델의 성능을 지속적으로 추적하고, 성능 저하 시 자동으로 재학습을 유도합니다.

---

## 📝 단계별 MLOps 파이프라인 구축 방안

### 1. 데이터 파이프라인 (Data Pipeline) 💧

고객사별로 독립적인 데이터 처리 파이프라인을 자동화하는 것이 핵심입니다.

-   **프로세스**: **Celery Beat**(스케줄러)를 사용해 매일 정해진 시간에 고객사별 데이터 수집 작업을 실행합니다. **Celery Worker**는 쇼피파이 등 이커머스 플랫폼 API를 호출하여 최신 데이터를 가져옵니다.
-   **데이터 저장 및 버전 관리**: 수집된 원본 데이터(Raw Data)와 가공된 학습용 데이터(Feature)는 **S3**에 고객사 ID와 날짜로 구분하여 저장합니다. (`s3://bucket/features/customer_id=123/date=2025-07-30/`) 이는 데이터 버저닝 역할을 하여 특정 시점의 학습 과정을 재현할 수 있게 합니다.

### 2. 모델 훈련 파이프라인 (CI/CD for Models) 🔄

코드가 아닌 '모델'을 위한 CI/CD(지속적 통합/지속적 배포)를 구축합니다.

-   **자동 훈련 트리거**: **Celery Beat**를 이용해 매주 또는 매월 자동으로 모델 재학습 파이프라인을 실행합니다.
-   **실험 관리**: 모든 모델 학습 과정은 **MLflow**에 자동으로 기록됩니다. (Git Commit, 데이터 버전, 파라미터, 성능 지표 등)
-   **모델 등록**: 학습된 모델 중 기존 운영 모델보다 성능이 좋은 모델이 나오면, **MLflow Model Registry**에 자동으로 새 버전을 등록하고 `Staging` 상태로 변경합니다.

### 3. 모델 배포 파이프라인 (Model Deployment) ☁️

MVP 단계에서는 실시간 추론보다 **배치(Batch) 예측** 방식이 훨씬 효율적이고 비용이 저렴합니다.

-   **프로세스**:
    1.  **MLflow**의 `Production` 단계에 등록된 모델을 불러오는 **Celery** 작업을 매일 실행합니다.
    2.  해당 모델로 고객사의 최신 사용자 데이터를 예측합니다.
    3.  예측 결과를 **PostgreSQL(RDS) 데이터베이스**에 저장합니다.
    4.  사용자 대시보드는 DB에 미리 저장된 예측 결과를 빠르고 간단하게 조회하여 보여줍니다.

### 4. 모델 모니터링 및 재학습 📈

-   **데이터 드리프트 (Data Drift) 감지**: 새로 들어오는 데이터의 통계적 분포가 학습 시점의 데이터와 크게 달라지는 것을 감지하여 모델 성능 저하를 예방합니다.
-   **성능 저하 (Concept Drift) 감지**: 모델의 예측 정확도가 시간이 지나면서 자연스럽게 떨어지는 현상을 모니터링합니다. 실제 정답을 알기 어려운 경우, 예측 점수의 분포 변화 같은 **대리 지표(Proxy Metric)**를 추적합니다.
-   **자동 재학습**: 모니터링 시스템에서 성능 저하가 감지되면, 자동으로 **모델 훈련 파이프라인(2번)**을 트리거하여 모델을 개선합니다.

---

이러한 MLOps 파이프라인은 소수의 인원으로도 수많은 고객사의 예측 모델을 안정적으로 운영하는 **'AI 분석 자동화 공장'**의 기반이 됩니다.