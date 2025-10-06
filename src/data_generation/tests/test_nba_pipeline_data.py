"""Tests for NBA pipeline data generation."""

from src.data_generation.nba_pipeline_data import StandardizedTrainingPipeline


def test_standardized_training_pipeline_init():
    """Test that StandardizedTrainingPipeline can be initialized."""
    # This is a basic test - we'd need actual player data for full testing
    # For now, just test that the class exists and can be imported
    assert StandardizedTrainingPipeline is not None


def test_adjusted_projected_line_method_exists():
    """Test that the adjusted_projected_line method exists."""
    # Test that the method exists on the class
    assert hasattr(StandardizedTrainingPipeline, 'adjusted_projected_line')
    assert hasattr(StandardizedTrainingPipeline, 'prepare_training_data')
    assert hasattr(StandardizedTrainingPipeline, 'get_career_stats')
