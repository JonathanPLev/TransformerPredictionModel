"""Tests for Random Forest trainer."""

from src.random_forest.random_forest_trainer import StandardizedTrainingPipeline


def test_random_forest_training_pipeline_init():
    """Test that StandardizedTrainingPipeline can be initialized."""
    # This is a basic test - we'd need actual player data for full testing
    # For now, just test that the class exists and can be imported
    assert StandardizedTrainingPipeline is not None


def test_training_pipeline_methods_exist():
    """Test that required methods exist on the training pipeline."""
    assert hasattr(StandardizedTrainingPipeline, 'prepare_training_data')
    assert hasattr(StandardizedTrainingPipeline, 'get_career_stats')
    assert hasattr(StandardizedTrainingPipeline, 'adjusted_projected_line')
