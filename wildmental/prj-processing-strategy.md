# MLOPS

## 구체화 방향
1. 학습 데이터 스키마 충분성
2. 기존 분석사례 정확도 범위
3. 시스템 최소 요건 (외부 / 내부)
4. 정상동작 바운더리 검사 및 추적 방법
  - 예측 이력 모니터링 & 인사이트 정리
  - 모델 및 프롬프트 버전 등에 따른 성능 관리
5. 예측 이상 발생 시 대응방법 버전 수립
6. 대응 방법별 자동화 방안
   1) 트리거 & 루틴 정의
     : 정상 예측 정확도 벗어날 경우 
       실행할 workflow 정의
   2) 의심되는 원인별 대응
     : Input 이상으로 의심될 경우
      - prompt Template 버전 변경 등
     : Output(모델성능) 이상으로 의심될 경우
      - 모델 변경, 모델 재학습 등

## 팀원 필요
- 열정을 가진 사람
- 분석에 열정을 가진 사람
- 자동화에 열정을 가진 사람