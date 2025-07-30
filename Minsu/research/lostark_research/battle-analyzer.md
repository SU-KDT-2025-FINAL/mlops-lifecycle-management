🤖 로스트아크 분석 프로젝트: MLOps/DataOps 도입 계획
1. 개요 (Overview)
본 문서는 '로스트아크 전투 분석 API' 기반 서비스의 지속 가능성과 확장성을 확보하기 위한 MLOps/DataOps 도입 계획을 정의합니다. 우리는 일회성 분석 도구가 아닌, 데이터가 자동으로 흐르고 스스로 개선하며 안정적으로 운영되는 살아있는 데이터 서비스를 구축하는 것을 목표로 합니다.

이를 위해 개발 초기 단계부터 '자동화 문화'를 정착시키고, 데이터 파이프라인의 신뢰성을 확보하는 것을 최우선 과제로 삼습니다.

2. MLOps의 역할과 범위 (Role & Scope)
우리 프로젝트에서 MLOps는 두 단계에 걸쳐 역할을 수행합니다.

Phase 1: 데이터 파이프라인의 안정화 (DataOps)
"신뢰할 수 있는 데이터가 흐르는 파이프라인을 구축한다."

머신러닝 모델이 없는 현 단계에서는 데이터 파이프라인의 자동화, 모니터링, 신뢰성 확보에 집중합니다.

데이터 수집 자동화: 로스트아크 API를 주기적으로 호출하고, 새로운 전투 기록을 안정적으로 수집합니다.

데이터 검증 및 처리: 수집된 데이터가 유효한지(예: 누락된 값, 형식 오류) 검증하고, Pandas를 이용해 분석 가능한 형태로 표준화합니다.

파이프라인 모니터링: 데이터 수집 및 처리 과정에 문제가 발생할 경우, 즉시 감지하고 알림을 보냅니다. (예: API 호출 실패, 데이터 처리 시간 급증)

Phase 2: 머신러닝 모델의 수명주기 관리 (Full MLOps)
"모델의 학습, 배포, 서빙, 모니터링을 자동화한다."

프로젝트가 고도화되어 '예상 DPS 예측 모델'이나 '사용자 클러스터링 모델' 등 머신러닝 모델을 도입하게 되면, 모델의 전체 수명주기를 관리합니다.

지속적 학습(CT): 새로운 데이터가 충분히 쌓이면 자동으로 모델을 재학습합니다.

지속적 배포(CD): 재학습된 모델의 성능이 기존 모델보다 우수할 경우, 자동으로 운영 환경에 배포합니다.

모델 성능 모니터링: 배포된 모델의 예측 성능이 저하되지 않는지 지속적으로 추적하고 관리합니다.

3. 목표 아키텍처 (Target Architecture)
우리 프로젝트가 지향하는 자동화된 데이터 분석 및 서빙 파이프라인의 개념도는 다음과 같습니다.

API Gateway ➔ [Backend] API 호출 ➔ [Data Lake/Warehouse] 원본 데이터 저장 ➔ [Processing] Pandas/Spark 데이터 처리 ➔ [DB] 분석용 데이터 저장 ➔ [Serving API] 분석 결과 제공 ➔ [Frontend] 데이터 시각화

CI/CD Pipeline (GitHub Actions): 코드 변경 시 자동으로 테스트, 빌드, 배포를 수행합니다.

Monitoring System (Prometheus/Grafana): 시스템의 모든 계층(서버, DB, 파이프라인)을 실시간으로 모니터링합니다.

4. 주요 기술 스택 (Key Tech Stack)
구분	기술	역할
Backend	Python (FastAPI/Django)	API 통신, 데이터 처리 로직 실행
Data Analysis	Pandas	핵심 데이터 처리 및 통계 분석
Database	PostgreSQL, Redis	분석 데이터 영구 저장, API 결과 캐싱
CI/CD	Docker, GitHub Actions	애플리케이션 컨테이너화, 배포 자동화
Orchestration	Kubernetes (EKS)	트래픽에 따른 서비스 확장성 확보
Monitoring	Prometheus, Grafana	시스템/파이프라인 상태 시각화 및 알림

Sheets로 내보내기
5. 단계별 추진 로드맵 (Phased Roadmap)
Step 1: 데이터 파이프라인 프로토타입 구축
로스트아크 API 호출 및 데이터 저장 기능 구현

Pandas를 이용한 핵심 분석 로직(평균 DPS 계산 등) 스크립트화

Step 2: CI/CD 및 기본 모니터링 도입
프로젝트의 Dockerfile 작성 및 컨테이너 환경에서 실행

GitHub Actions를 통해 Git Push 시 자동으로 테스트 및 배포 파이프라인 구축

로그 수집 및 기본 에러 알림 시스템 연동

Step 3: 파이프라인 고도화 및 ML 모델 도입 준비
데이터 검증(Data Validation) 로직 추가

(필요시) Scikit-learn을 이용한 예측 모델 프로토타입 개발 및 학습 파이프라인 구상

6. 팀원과의 약속 (Our Commitment)
Automate Everything: 반복적인 모든 작업은 자동화를 지향합니다.

Monitor Everything: 우리의 작업이 운영 환경에서 어떻게 동작하는지 항상 주시합니다.

Test Everything: 우리의 코드가 예상대로 동작함을 테스트로 증명합니다.

Data is King: 모든 의사결정은 데이터를 기반으로 합니다.