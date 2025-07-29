# MLOps 스터디 가이드

## 📚 목차
1. [MLOps 개념](#mlops-개념)
2. [MLOps 라이프사이클](#mlops-라이프사이클)
3. [핵심 도구 및 기술](#핵심-도구-및-기술)
4. [학습 로드맵](#학습-로드맵)
5. [실습 프로젝트](#실습-프로젝트)

---

## 🎯 MLOps 개념

### MLOps란 무엇인가?
MLOps(Machine Learning Operations)는 머신러닝 모델의 개발, 배포, 운영을 자동화하고 표준화하는 실무 방법론입니다.

### 왜 MLOps가 필요한가?
- **재현 가능성**: 실험과 결과를 언제든 재현할 수 있어야 함
- **확장성**: 모델을 프로덕션 환경에서 안정적으로 운영
- **협업**: 데이터 사이언티스트, 엔지니어, 운영팀 간의 원활한 협업
- **자동화**: 반복적인 작업의 자동화로 효율성 증대

---

## 🔄 MLOps 라이프사이클

### 1. 데이터 수집 및 준비 (Data Collection & Preparation)
- 데이터 수집 파이프라인 구축
- 데이터 검증 및 품질 관리
- 데이터 버전 관리 (DVC 활용)

### 2. 모델 개발 및 실험 (Model Development & Experimentation)
- 피처 엔지니어링
- 모델 학습 및 튜닝
- 실험 추적 및 관리 (MLflow 활용)

### 3. 모델 검증 및 평가 (Model Validation & Evaluation)
- 모델 성능 평가
- A/B 테스트
- 모델 승인 프로세스

### 4. 모델 배포 (Model Deployment)
- CI/CD 파이프라인 구축
- 컨테이너화 (Docker)
- 서빙 인프라 구축

### 5. 모델 모니터링 (Model Monitoring)
- 성능 모니터링
- 데이터 드리프트 탐지
- 모델 재학습 트리거

---

## 🛠 핵심 도구 및 기술

### 데이터 관리
- **DVC (Data Version Control)**: 데이터 버전 관리
- **Apache Airflow**: 데이터 파이프라인 오케스트레이션
- **Great Expectations**: 데이터 품질 검증

### 실험 추적
- **MLflow**: 실험 추적 및 모델 관리
- **Weights & Biases**: 실험 시각화 및 협업
- **Neptune**: 메타데이터 저장소

### 모델 배포
- **Docker**: 컨테이너화
- **Kubernetes**: 오케스트레이션
- **FastAPI**: 모델 서빙 API
- **TensorFlow Serving**: TensorFlow 모델 전용 서빙

### CI/CD
- **GitHub Actions**: 자동화 워크플로우
- **Jenkins**: 지속적 통합/배포
- **GitLab CI/CD**: 통합 DevOps 플랫폼

### 모니터링
- **Prometheus**: 메트릭 수집
- **Grafana**: 시각화 대시보드
- **Evidently AI**: 모델 및 데이터 모니터링

---

## 📖 학습 로드맵

### 1단계: 기초 개념 (1-2주)
- [ ] MLOps 개념 및 필요성 이해
- [ ] ML 프로젝트 라이프사이클 학습
- [ ] 버전 관리 시스템 (Git) 숙달

### 2단계: 데이터 관리 (2-3주)
- [ ] DVC 설치 및 기본 사용법
- [ ] 데이터 파이프라인 설계
- [ ] 데이터 품질 검증 도구 학습

### 3단계: 실험 추적 (2-3주)
- [ ] MLflow 설치 및 설정
- [ ] 실험 로깅 및 추적
- [ ] 모델 레지스트리 활용

### 4단계: 모델 배포 (3-4주)
- [ ] Docker 기초 및 컨테이너화
- [ ] FastAPI를 활용한 모델 서빙
- [ ] CI/CD 파이프라인 구축

### 5단계: 모니터링 및 운영 (2-3주)
- [ ] 모델 성능 모니터링
- [ ] 로깅 및 알림 시스템
- [ ] 모델 재학습 자동화

---

## 🚀 실습 프로젝트

### 프로젝트 1: 기본 MLOps 파이프라인
**목표**: 간단한 머신러닝 모델의 전체 라이프사이클 구현

**구성 요소**:
- 데이터 수집 및 전처리
- 모델 학습 및 평가
- MLflow를 활용한 실험 추적
- FastAPI를 활용한 모델 서빙
- Docker 컨테이너화

### 프로젝트 2: 고급 MLOps 시스템
**목표**: 프로덕션 레벨의 MLOps 시스템 구축

**구성 요소**:
- CI/CD 파이프라인 (GitHub Actions)
- 모델 모니터링 시스템
- 자동 재학습 트리거
- A/B 테스트 프레임워크

---

## 📝 학습 체크리스트

### 이론 학습
- [ ] MLOps 개념 및 원칙 이해
- [ ] DevOps vs MLOps 차이점 파악
- [ ] 모델 라이프사이클 관리 방법 학습

### 실무 도구
- [ ] Git/GitHub 활용 능력
- [ ] Docker 컨테이너 기술
- [ ] Python 개발 환경 설정
- [ ] 클라우드 플랫폼 (AWS/GCP/Azure) 기초

### 핵심 실습
- [ ] DVC를 활용한 데이터 버전 관리
- [ ] MLflow를 활용한 실험 추적
- [ ] CI/CD 파이프라인 구축
- [ ] 모델 모니터링 시스템 구현

---

## 📚 추천 학습 자료

### 온라인 강의
- [MLOps Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
- [Practical MLOps (O'Reilly)](https://www.oreilly.com/library/view/practical-mlops/9781098103002/)

### 도서
- "Building Machine Learning Pipelines" - Hannes Hapke, Catherine Nelson
- "Introducing MLOps" - Mark Treveil, Nicolas Omont

### 블로그 및 문서
- [MLOps.org](https://mlops.org/)
- [Google Cloud MLOps 가이드](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS MLOps 베스트 프랙티스](https://aws.amazon.com/machine-learning/mlops/)

---

## 💡 다음 스터디 계획

1. **1주차**: MLOps 개념 및 환경 설정
2. **2주차**: DVC를 활용한 데이터 관리
3. **3주차**: MLflow 실험 추적 시스템
4. **4주차**: Docker 및 컨테이너화
5. **5주차**: CI/CD 파이프라인 구축
6. **6주차**: 모델 모니터링 및 운영

---

## 📞 스터디 진행 방식

- **주간 미팅**: 매주 진행 상황 공유 및 토론
- **실습 과제**: 각 주차별 실습 과제 수행
- **프로젝트 발표**: 스터디 종료 시 최종 프로젝트 발표
- **코드 리뷰**: GitHub을 통한 코드 리뷰 및 피드백

---

*이 문서는 MLOps 스터디의 가이드라인이며, 진행 상황에 따라 조정될 수 있습니다.* 