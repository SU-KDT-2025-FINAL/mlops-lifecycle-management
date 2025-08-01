## 온라인 쇼핑몰 기반 솔루션 전제로 프로젝트 아이디어 피드백
1. 고객 : "개인 쇼핑몰" 운영 사업체
  -> 도메인 타게팅은 시장 규모, 잠재 고객 규모에서 결정됨  
2. 지불 의사가 있고, 지불 역량이 있는 고객
  - 분석사례?
    : 돈 되는 사례
---
## 전략
1) 잠재고객에 대해서 날카롭게 파고들기
2) 확장할 수 있는 영역 찾아보기
*1, 2) 한번에 처음부터 통합하 싶은 욕심*
   -> 기존 사례에서 유사 필드, 유사 데이터 패턴을 보이는
      여러 도메인이 있는가? (확장성을 미리 고려 가능)
---
### 시스템적 적용 방법
1. (데이터 수신) 기존 고객 웹사이트에 이식 가능한 Data 수신 구조를 고민해야 함
- 고객 기본 정보 => 우리 서버로
- 고객의 활동 정보 => 주기적 전송
- 고객의 외부 플랫폼 정보 => 구글/카카오/금융 등의 활동 정보

2. (분석 결과 제공)분석 된 결과를 고객 웹사이트에 적용가능한 방법을 고민해야 함 - 구체적인 사이트를 타겟으로 하면 쉽다
- 더미 사이트를 만들어서 적용하면 더 쉽다

---
### 리서치 방향
1. 대형 온라인 몰 (여러개의 몰 통합 운영 지원 도구)
  - 고객 광고 집행 등 -> 대시보드로 고객 인사이트 제공
2. 각 몰에서 (중견 이상) 수행하는 분석
  - Growth Hack 분야 : CRM 에서 진화해서 고객 활동지표 분석
    => 자동화 가능한 영역 발라내기
---
### 리서치할 내용 및 진행 순서
1. 수집해야 할 데이터 명세 확인 (분석사례에서)
2. 더미 데이터 스키마 설계 및 구현 (공공데이터에서)
  - 고객 데이터
  - 고객 활동 데이터
3. 데이터 수집/분석 구현  
4. 더미 몰에 적용 여부 결정
  1) 분석 flow 가 구축된 서비스에 통합되어 있거나
  2) 주기적으로 분석 대상 데이터를 수신해서 분석
5. (분석 개선 자동화) 분석 과정에서 예측 정확도가
   정상범위 이하로 떨어질 경우 정확도 재확보를 위한 flow 구현
6. (향후진행) (SI 구축 자동화) 분석 솔루션 고객 몰 자동적용 flow 구현
---
### 분석 적용 결과를 Actionable 하게 제시 or 시스템 적용 형태로 제시하기
1) 고객 선호도 분석
  -> AS IS / TO BE
     (분석없이 했던 동작)
      => (분석후에 할 수 있는 동작)
2) 고객 이탈위험도 분석
  -> (분석없이 했던 동작)
      => (팝업으로 쿠폰 or 할인 정보 or 관심 상품)