# Nike 고객 이탈 예측 - 상세 스키마 및 컬럼 설명

##  핵심 테이블 스키마 (컬럼별 상세)

### 1. 고객 테이블 (customers)
| 컬럼명 | 데이터 타입 | 설명 | 예시 값 |
|--------|-------------|------|---------|
| `customer_id` | VARCHAR(50) | 고객 고유 ID (Primary Key) | "CUST_001234" |
| `email` | VARCHAR(255) | 고객 이메일 주소 | "john.doe@email.com" |
| `registration_date` | DATE | 고객 가입일 | "2022-03-15" |
| `country` | VARCHAR(50) | 거주 국가 | "US", "KR", "JP" |

### 2. 주문 테이블 (orders)
| 컬럼명 | 데이터 타입 | 설명 | 예시 값 |
|--------|-------------|------|---------|
| `order_id` | VARCHAR(50) | 주문 고유 ID (Primary Key) | "ORD_20230415_001" |
| `customer_id` | VARCHAR(50) | 주문한 고객 ID (Foreign Key) | "CUST_001234" |
| `order_date` | DATE | 주문 발생 날짜 | "2023-04-15" |
| `total_amount` | DECIMAL(10,2) | 주문 총액 (USD) | 129.99 |
| `product_category` | VARCHAR(50) | 상품 카테고리 | "shoes", "apparel", "equipment" |

### 3. 앱 사용 테이블 (app_usage)
| 컬럼명 | 데이터 타입 | 설명 | 예시 값 |
|--------|-------------|------|---------|
| `customer_id` | VARCHAR(50) | 고객 ID (Primary Key 1) | "CUST_001234" |
| `usage_date` | DATE | 앱 사용 날짜 (Primary Key 2) | "2023-04-15" |
| `session_minutes` | INTEGER | 앱 사용 시간 (분) | 25 |
| `workouts_completed` | INTEGER | 완료한 운동 수 (Nike Training Club) | 2 |

---

##  분석용 계산 컬럼들

### SQL로 생성되는 피처 컬럼들
| 피처명 | 계산 방법 | 설명 | 예측 기여도 |
|--------|-----------|------|-------------|
| `total_orders` | `COUNT(order_id)` | 총 주문 횟수 | ⭐⭐⭐⭐ |
| `total_spent` | `SUM(total_amount)` | 총 구매 금액 | ⭐⭐⭐ |
| `avg_order_value` | `AVG(total_amount)` | 평균 주문 금액 | ⭐⭐⭐ |
| `days_since_last_order` | `DATEDIFF(CURRENT_DATE, MAX(order_date))` | 마지막 주문 이후 일수 | ⭐⭐⭐⭐⭐ |
| `avg_days_between_orders` | `DATEDIFF(MAX, MIN) / (COUNT - 1)` | 평균 주문 주기 | ⭐⭐ |
| `workouts_last_90d` | `SUM(workouts_completed)` | 최근 90일 운동 완료 수 | ⭐⭐⭐⭐ |
| `days_since_app_use` | `DATEDIFF(CURRENT_DATE, MAX(usage_date))` | 마지막 앱 사용 이후 일수 | ⭐⭐⭐ |
| `is_churned` | `CASE WHEN days_since_last_order > 60 THEN 1 ELSE 0` | 이탈 여부 (타겟 변수) | 🎯 |

---

##  실제 데이터 예시

### customers 테이블 샘플
```sql
customer_id    | email                | registration_date | country
CUST_001234   | john.doe@email.com   | 2022-03-15       | US
CUST_001235   | jane.kim@email.com   | 2022-05-20       | KR  
CUST_001236   | mike.tanaka@email.com| 2021-12-10       | JP
```

### orders 테이블 샘플
```sql
order_id           | customer_id | order_date | total_amount | product_category
ORD_20230415_001  | CUST_001234 | 2023-04-15 | 129.99      | shoes
ORD_20230420_002  | CUST_001234 | 2023-04-20 | 89.99       | apparel
ORD_20230301_003  | CUST_001235 | 2023-03-01 | 199.99      | shoes
```

### app_usage 테이블 샘플
```sql
customer_id | usage_date | session_minutes | workouts_completed
CUST_001234 | 2023-04-15 | 25             | 2
CUST_001234 | 2023-04-16 | 18             | 1
CUST_001235 | 2023-04-15 | 45             | 3
```

### 최종 분석 결과 샘플
```sql
customer_id | total_orders | total_spent | days_since_last_order | workouts_last_90d | is_churned
CUST_001234 | 5           | 649.95      | 15                   | 25               | 0
CUST_001235 | 2           | 299.98      | 45                   | 18               | 0  
CUST_001236 | 1           | 159.99      | 85                   | 0                | 1
```

---

##  우리 프로젝트용 최소 스키마

### SMB용 간소화 버전 (2개 테이블)
```sql
-- 고객 테이블
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255),
    created_at DATE
);

-- 주문 테이블
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    created_at DATE,
    total_price DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

### 핵심 분석 쿼리 (4개 피처만)
```sql
SELECT 
    customer_id,
    COUNT(*) as order_count,                                    -- 주문 횟수
    SUM(total_price) as total_spent,                           -- 총 구매액
    MAX(created_at) as last_order_date,                        -- 마지막 주문일
    DATEDIFF(CURRENT_DATE, MAX(created_at)) as days_since_last_order, -- 마지막 주문 이후 일수
    
    -- 이탈 여부 (30일 기준으로 단순화)
    CASE WHEN DATEDIFF(CURRENT_DATE, MAX(created_at)) > 30 THEN 1 ELSE 0 END as is_churned
    
FROM orders 
GROUP BY customer_id;
```

이렇게 **최소 2개 테이블, 4개 핵심 피처**만으로도 효과적인 이탈 예측이 가능합니다!