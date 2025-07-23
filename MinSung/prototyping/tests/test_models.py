"""Tests for model training and evaluation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.config import Config


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.load_config()
    
    @pytest.fixture
    def trainer(self, config):
        """Create ModelTrainer instance."""
        return ModelTrainer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create binary target
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y
    
    def test_prepare_features(self, trainer, sample_data):
        """Test feature preparation."""
        X, y = sample_data
        
        X_processed, y_processed = trainer.prepare_features(X)
        
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y_processed, pd.Series)
        assert len(X_processed) == len(X)
        assert len(y_processed) == len(y)
    
    def test_train_model(self, trainer, sample_data):
        """Test model training."""
        X, y = sample_data
        
        results = trainer.train_model(X, y)
        
        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert "test_precision" in results
        assert "test_recall" in results
        assert "test_f1" in results
        assert "test_roc_auc" in results
        assert "cv_accuracy_mean" in results
        assert "cv_f1_mean" in results
        assert "cv_roc_auc_mean" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert "feature_importance" in results
    
    def test_hyperparameter_tuning(self, trainer, sample_data):
        """Test hyperparameter tuning."""
        X, y = sample_data
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        results = trainer.hyperparameter_tuning(X, y, param_grid)
        
        assert "best_params" in results
        assert "best_score" in results
        assert "cv_results" in results
    
    def test_save_model(self, trainer, sample_data, tmp_path):
        """Test model saving."""
        X, y = sample_data
        
        # Train model first
        trainer.train_model(X, y)
        
        # Save model
        model_path = trainer.save_model("test_model", "v1.0")
        
        assert model_path is not None
        assert isinstance(model_path, str)


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.load_config()
    
    @pytest.fixture
    def evaluator(self, config):
        """Create ModelEvaluator instance."""
        return ModelEvaluator(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create binary target
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    def test_calculate_basic_metrics(self, evaluator, sample_data, trained_model):
        """Test basic metrics calculation."""
        X, y = sample_data
        
        y_pred = trained_model.predict(X)
        y_proba = trained_model.predict_proba(X)[:, 1]
        
        metrics = evaluator._calculate_basic_metrics(y, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        
        # Check metric values are reasonable
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"
    
    def test_calculate_advanced_metrics(self, evaluator, sample_data, trained_model):
        """Test advanced metrics calculation."""
        X, y = sample_data
        
        y_pred = trained_model.predict(X)
        y_proba = trained_model.predict_proba(X)[:, 1]
        
        advanced_metrics = evaluator._calculate_advanced_metrics(y, y_pred, y_proba)
        
        assert "confusion_matrix" in advanced_metrics
        assert "classification_report" in advanced_metrics
        assert "roc_curve" in advanced_metrics
        assert "precision_recall_curve" in advanced_metrics
        assert "average_precision" in advanced_metrics
    
    def test_cross_validate_model(self, evaluator, sample_data, trained_model):
        """Test cross-validation."""
        X, y = sample_data
        
        cv_results = evaluator._cross_validate_model(trained_model, X, y)
        
        assert "accuracy_mean" in cv_results
        assert "accuracy_std" in cv_results
        assert "f1_mean" in cv_results
        assert "f1_std" in cv_results
        assert "roc_auc_mean" in cv_results
        assert "roc_auc_std" in cv_results
    
    def test_evaluate_model(self, evaluator, sample_data, trained_model):
        """Test complete model evaluation."""
        X, y = sample_data
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        evaluation_results = evaluator.evaluate_model(
            trained_model, X_test, y_test, X_train, y_train
        )
        
        assert "basic_metrics" in evaluation_results
        assert "advanced_metrics" in evaluation_results
        assert "cross_validation" in evaluation_results
        assert "plots" in evaluation_results
        assert "n_samples" in evaluation_results
        assert "n_features" in evaluation_results
        assert "class_distribution" in evaluation_results
    
    def test_generate_evaluation_plots(self, evaluator, sample_data, trained_model):
        """Test plot generation."""
        X, y = sample_data
        
        y_pred = trained_model.predict(X)
        y_proba = trained_model.predict_proba(X)[:, 1]
        
        plots = evaluator._generate_evaluation_plots(y, y_pred, y_proba)
        
        assert "confusion_matrix" in plots
        assert "roc_curve" in plots
        assert "precision_recall_curve" in plots
        
        # Check that plot files exist
        for plot_path in plots.values():
            assert isinstance(plot_path, str)
    
    def test_save_evaluation_report(self, evaluator, sample_data, trained_model, tmp_path):
        """Test evaluation report saving."""
        X, y = sample_data
        
        # Create evaluation results
        evaluation_results = {
            "basic_metrics": {
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.85,
                "f1": 0.84,
                "roc_auc": 0.88,
            },
            "cross_validation": {
                "accuracy_mean": 0.83,
                "accuracy_std": 0.05,
                "f1_mean": 0.82,
                "f1_std": 0.06,
                "roc_auc_mean": 0.86,
                "roc_auc_std": 0.04,
            },
            "n_samples": 100,
            "n_features": 5,
            "class_distribution": {0: 50, 1: 50},
            "plots": {"confusion_matrix": "test.png"},
        }
        
        report_path = evaluator.save_evaluation_report(
            evaluation_results, "test_model", "v1.0"
        )
        
        assert report_path is not None
        assert isinstance(report_path, str)
    
    def test_compare_models(self, evaluator):
        """Test model comparison."""
        # Create mock evaluation results
        model_results = [
            {
                "model_name": "model1",
                "version": "v1.0",
                "basic_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.84,
                    "recall": 0.85,
                    "f1": 0.84,
                    "roc_auc": 0.88,
                },
                "cross_validation": {
                    "accuracy_mean": 0.83,
                    "f1_mean": 0.82,
                    "roc_auc_mean": 0.86,
                },
                "n_samples": 100,
                "n_features": 5,
            },
            {
                "model_name": "model2",
                "version": "v1.0",
                "basic_metrics": {
                    "accuracy": 0.87,
                    "precision": 0.86,
                    "recall": 0.87,
                    "f1": 0.86,
                    "roc_auc": 0.90,
                },
                "cross_validation": {
                    "accuracy_mean": 0.85,
                    "f1_mean": 0.84,
                    "roc_auc_mean": 0.88,
                },
                "n_samples": 100,
                "n_features": 5,
            },
        ]
        
        comparison_df = evaluator.compare_models(model_results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert "model_name" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "f1" in comparison_df.columns
        assert "roc_auc" in comparison_df.columns


class TestModelIntegration:
    """Integration tests for model training and evaluation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.load_config()
    
    def test_training_and_evaluation_pipeline(self, config):
        """Test complete training and evaluation pipeline."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        trainer = ModelTrainer(config)
        training_results = trainer.train_model(X_train, y_train)
        
        # Evaluate model
        evaluator = ModelEvaluator(config)
        evaluation_results = evaluator.evaluate_model(
            trainer.model, X_test, y_test, X_train, y_train
        )
        
        # Check that both training and evaluation work together
        assert "test_accuracy" in training_results
        assert "basic_metrics" in evaluation_results
        
        # Check that metrics are consistent
        assert abs(training_results["test_accuracy"] - 
                  evaluation_results["basic_metrics"]["accuracy"]) < 0.01 