"""Tests for Random Forest trainer."""

from random_forest.random_forest_trainer import RandomForestTrainer


def test_random_forest_training_pipeline_init():
    """Test that StandardizedTrainingPipeline can be initialized."""
    # This is a basic test - we'd need actual player data for full testing
    # For now, just test that the class exists and can be imported
    assert RandomForestTrainer is not None


def test_training_pipeline_methods_exist():
    """Test that required methods exist on the training pipeline."""
    assert hasattr(RandomForestTrainer, 'train_model')
    assert hasattr(RandomForestTrainer, 'cross_validate_with_models')
    assert hasattr(RandomForestTrainer, 'save_model_with_metadata')
