"""Tests for CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cfdna.cli import main


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "cfDNA Epigenomics Mini" in result.output
    assert "simulate" in result.output
    assert "train" in result.output
    assert "eval" in result.output


def test_cli_version(runner):
    """Test CLI version command."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_simulate_help(runner):
    """Test simulate subcommand help."""
    result = runner.invoke(main, ["simulate", "--help"])
    assert result.exit_code == 0
    assert "Simulate synthetic cfDNA" in result.output


def test_train_help(runner):
    """Test train subcommand help."""
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train machine learning models" in result.output


def test_eval_help(runner):
    """Test eval subcommand help."""
    result = runner.invoke(main, ["eval", "--help"])
    assert result.exit_code == 0
    assert "Evaluate trained models" in result.output


def test_report_help(runner):
    """Test report subcommand help."""
    result = runner.invoke(main, ["report", "--help"])
    assert result.exit_code == 0
    assert "Generate HTML report" in result.output


def test_smoke_help(runner):
    """Test smoke subcommand help."""
    result = runner.invoke(main, ["smoke", "--help"])
    assert result.exit_code == 0
    assert "Run end-to-end smoke test" in result.output


def test_simulate_minimal(runner):
    """Test minimal simulate command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        result = runner.invoke(main, [
            "simulate",
            "--out", str(temp_path / "data"),
            "--seed", "42"
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_lines = result.output.strip().split('\n')
        json_output = '\n'.join(line for line in output_lines if line.startswith('{'))
        
        if json_output:
            data = json.loads(json_output)
            assert data["status"] == "success"
            assert "run_id" in data
            assert "stats" in data


def test_features_missing_data_dir(runner):
    """Test features command with missing data directory."""
    result = runner.invoke(main, [
        "features",
        "--data-dir", "/nonexistent",
        "--out", "/tmp/test",
        "--seed", "42"
    ])
    
    assert result.exit_code != 0


def test_train_missing_features_dir(runner):
    """Test train command with missing features directory.""" 
    result = runner.invoke(main, [
        "train",
        "--features-dir", "/nonexistent",
        "--out", "/tmp/test",
        "--models", "logistic",
        "--seed", "42"
    ])
    
    assert result.exit_code != 0


def test_eval_missing_models_dir(runner):
    """Test eval command with missing models directory."""
    result = runner.invoke(main, [
        "eval", 
        "--models-dir", "/nonexistent",
        "--out", "/tmp/test",
        "--seed", "42"
    ])
    
    assert result.exit_code != 0


def test_report_missing_results_dir(runner):
    """Test report command with missing results directory."""
    result = runner.invoke(main, [
        "report",
        "--results-dir", "/nonexistent", 
        "--out", "/tmp/test.html"
    ])
    
    assert result.exit_code != 0


@patch('cfdna.cli.time.time')
def test_smoke_timeout_warning(mock_time, runner):
    """Test smoke command timeout warning."""
    # Mock time to simulate long runtime
    mock_time.side_effect = [0, 400]  # 400 seconds = > 5 minutes
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cfdna.cli.simulate_dataset'), \
             patch('cfdna.cli.prepare_features'), \
             patch('cfdna.cli.train_models'), \
             patch('cfdna.cli.evaluate_models'), \
             patch('cfdna.cli.generate_report'):
            
            result = runner.invoke(main, ["smoke", "--seed", "42"])
            
            # Should complete but warn about time
            assert result.exit_code == 0
            output_lines = result.output.strip().split('\n')
            json_output = '\n'.join(line for line in output_lines if line.startswith('{'))
            
            if json_output:
                data = json.loads(json_output)
                assert data["elapsed_time_seconds"] == 400


def test_logging_setup(runner):
    """Test that logging setup works with verbose flag."""
    result = runner.invoke(main, ["-v", "--help"])
    assert result.exit_code == 0


def test_device_options(runner):
    """Test device options in commands."""
    # Test simulate with device
    result = runner.invoke(main, ["simulate", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output
    
    # Test train with device
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output
    
    # Test smoke with device
    result = runner.invoke(main, ["smoke", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output


def test_multiple_models_option(runner):
    """Test multiple models option in train command."""
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    # The help should show that --models can be used multiple times
    assert "--models" in result.output