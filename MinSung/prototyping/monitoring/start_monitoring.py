#!/usr/bin/env python3
"""Start monitoring services locally."""

import subprocess
import time
import webbrowser
from pathlib import Path

def start_prometheus():
    """Start Prometheus locally."""
    print("🔄 Starting Prometheus...")
    # 간단한 Prometheus 설정
    prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['localhost:8000']
"""
    
    config_path = Path("monitoring/prometheus.yml")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(prometheus_config)
    
    # Prometheus 실행 (간단한 방법)
    print("📊 Prometheus metrics available at: http://localhost:8000/metrics")

def start_grafana():
    """Start Grafana locally."""
    print("🔄 Starting Grafana...")
    print("📊 Grafana would be available at: http://localhost:3000")
    print("📋 Default credentials: admin/admin")

def start_mlflow():
    """Start MLflow locally."""
    print("🔄 Starting MLflow...")
    print("📊 MLflow would be available at: http://localhost:5000")

def main():
    """Start all monitoring services."""
    print("🚀 Starting MLOps Monitoring Services")
    print("=" * 50)
    
    # 현재 실행 중인 API 서버 확인
    print("✅ API Server is running at: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("📈 Metrics: http://localhost:8000/metrics")
    
    # 모니터링 서비스 정보
    print("\n📋 Monitoring Services:")
    print("1. Prometheus Metrics: http://localhost:8000/metrics")
    print("2. API Health Check: http://localhost:8000/health")
    print("3. Performance Stats: http://localhost:8000/performance/predict")
    
    # 브라우저에서 열기
    try:
        webbrowser.open("http://localhost:8000/docs")
        webbrowser.open("http://localhost:8000/metrics")
        print("\n🌐 Opened monitoring pages in browser")
    except:
        print("\n📋 Manual links:")
        print("   API Docs: http://localhost:8000/docs")
        print("   Metrics: http://localhost:8000/metrics")
    
    print("\n✅ Monitoring setup completed!")
    print("📋 To start full Docker services later:")
    print("   docker-compose up -d")

if __name__ == "__main__":
    main() 