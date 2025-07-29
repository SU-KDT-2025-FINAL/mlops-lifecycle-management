# MLOps 모범 사례: 성공적인 ML 시스템 구축

## 📋 목차
1. [코드 품질 관리](#코드-품질-관리)
2. [데이터 관리](#데이터-관리)
3. [실험 관리](#실험-관리)
4. [모델 배포](#모델-배포)
5. [모니터링](#모니터링)
6. [보안](#보안)

---

## 🎯 코드 품질 관리

### 1. 타입 힌팅과 문서화
```python
from typing import List, Dict, Optional, Tuple
import logging

def preprocess_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    scaler: Optional[object] = None
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    특성 데이터를 전처리합니다.
    
    Args:
        data: 입력 데이터프레임
        feature_columns: 사용할 특성 컬럼 목록
        target_column: 타겟 컬럼명
        scaler: 스케일러 객체 (None이면 새로 생성)
        
    Returns:
        전처리된 특성, 타겟 데이터, 스케일러 객체
        
    Raises:
        ValueError: 필수 컬럼이 없는 경우
    """
    logger = logging.getLogger(__name__)
    
    # 입력 검증
    missing_columns = set(feature_columns + [target_column]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # 특성 전처리
    features = data[feature_columns].values
    targets = data[target_column].values
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    logger.info(f"Preprocessed {len(features)} samples with {len(feature_columns)} features")
    
    return features, targets, scaler
```

### 2. 에러 처리 및 로깅
```python
import logging
from functools import wraps
from typing import Callable, Any

def handle_ml_errors(func: Callable) -> Callable:
    """ML 함수의 에러를 처리하는 데코레이터"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@handle_ml_errors
def train_model(X_train: np.ndarray, y_train: np.ndarray, model_params: Dict) -> object:
    """모델 학습 함수"""
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model
```

---

## 💾 데이터 관리

### 1. DVC를 사용한 데이터 버전 관리
```bash
# 데이터 추가
dvc add data/raw/dataset.csv

# 파이프라인 생성
dvc run -n preprocess \
    -d data/raw/dataset.csv \
    -d src/preprocess.py \
    -o data/processed/ \
    python src/preprocess.py

# 원격 저장소 설정
dvc remote add -d myremote s3://my-bucket/dvc
dvc push
```

### 2. 데이터 품질 검증
```python
import great_expectations as ge

def validate_data_quality(df: pd.DataFrame) -> bool:
    """데이터 품질 검증"""
    context = ge.get_context()
    
    # 검증 규칙 정의
    validator = context.get_validator(
        batch_request=df,
        expectation_suite_name="data_quality_suite"
    )
    
    # 검증 실행
    validator.expect_column_values_to_not_be_null("target")
    validator.expect_column_values_to_be_between("age", 0, 120)
    validator.expect_column_values_to_be_unique("id")
    
    # 결과 확인
    results = validator.validate()
    return results.success
```

---

## 🧪 실험 관리

### 1. MLflow를 사용한 실험 추적
```python
import mlflow
import mlflow.sklearn

def run_experiment(X_train: np.ndarray, y_train: np.ndarray, params: Dict) -> Tuple[object, float]:
    """MLflow를 사용한 실험 실행"""
    with mlflow.start_run():
        # 파라미터 로깅
        mlflow.log_params(params)
        
        # 모델 학습
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # 메트릭 로깅
        mlflow.log_metric("cv_accuracy", scores.mean())
        mlflow.log_metric("cv_std", scores.std())
        
        # 모델 저장
        mlflow.sklearn.log_model(model, "model")
        
        return model, scores.mean()
```

### 2. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """하이퍼파라미터 튜닝"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_
```

---

## 🚀 모델 배포

### 1. FastAPI를 사용한 모델 서빙
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 모델 로드
        model = joblib.load("model.pkl")
        
        # 예측
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            model_version="v1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Docker 컨테이너화
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 모니터링

### 1. 데이터 드리프트 탐지
```python
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
    """데이터 드리프트 탐지"""
    # Evidently AI 사용
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=ColumnMapping(
            target='target',
            numerical_features=['feature1', 'feature2']
        )
    )
    
    return data_drift_report.as_dict()
```

### 2. 모델 성능 모니터링
```python
import logging
from datetime import datetime
from typing import Dict, Any

def log_prediction(features: List[float], prediction: float, actual: Optional[float] = None) -> None:
    """예측 결과 로깅"""
    logger = logging.getLogger(__name__)
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "actual": actual,
        "model_version": "v1.0.0"
    }
    
    logger.info(log_data)

def calculate_performance_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """성능 메트릭 계산"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions, average='weighted'),
        "recall": recall_score(actuals, predictions, average='weighted'),
        "f1_score": f1_score(actuals, predictions, average='weighted')
    }
```

---

## 🔒 보안

### 1. 환경 변수 관리
```python
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 설정
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
```

### 2. API 인증
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """토큰 검증"""
    token = credentials.credentials
    # 토큰 검증 로직
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    # 예측 로직
    pass
```

---

## 📋 체크리스트

### 프로젝트 설정
- [ ] Python 3.10+ 사용
- [ ] 타입 힌팅 적용
- [ ] Google 스타일 Docstring 작성
- [ ] 로깅 설정
- [ ] 환경 변수 관리

### 데이터 관리
- [ ] DVC 설정
- [ ] 데이터 품질 검증
- [ ] 데이터 버전 관리
- [ ] 파이프라인 자동화

### 실험 관리
- [ ] MLflow 설정
- [ ] 실험 추적
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 버전 관리

### 모델 배포
- [ ] FastAPI 구현
- [ ] Docker 컨테이너화
- [ ] API 문서화
- [ ] 헬스 체크

### 모니터링
- [ ] 데이터 드리프트 탐지
- [ ] 성능 모니터링
- [ ] 로그 분석
- [ ] 알림 설정

### 보안
- [ ] 환경 변수 관리
- [ ] API 인증
- [ ] 데이터 암호화
- [ ] 접근 제어

---

## 🎯 결론

MLOps 모범 사례를 따르면 안정적이고 확장 가능한 ML 시스템을 구축할 수 있습니다. 핵심은 자동화, 모니터링, 그리고 지속적인 개선입니다. 