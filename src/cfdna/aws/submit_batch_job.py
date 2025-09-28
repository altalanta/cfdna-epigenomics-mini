"""Submit cfDNA cancer detection job to AWS Batch."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

def check_aws_credentials() -> bool:
    """Check if AWS credentials are available.
    
    Returns:
        True if credentials are found, False otherwise
    """
    # Check for AWS credentials in environment variables
    aws_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    env_creds = all(var in os.environ for var in aws_env_vars)
    
    # Check for AWS profile
    aws_profile = os.environ.get("AWS_PROFILE")
    
    # Check for IAM role (would be available in EC2/ECS)
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        boto3_creds = credentials is not None
    except ImportError:
        boto3_creds = False
    except Exception:
        boto3_creds = False
    
    return env_creds or aws_profile or boto3_creds


def generate_batch_command(job_name: str, config_path: str, model_type: str, 
                          output_path: str, experiment_id: str = "default") -> str:
    """Generate AWS Batch CLI command.
    
    Args:
        job_name: Name for the batch job
        config_path: S3 path to config file
        model_type: Type of model to train
        output_path: S3 path for outputs
        experiment_id: Experiment identifier
        
    Returns:
        AWS CLI command string
    """
    job_def = "cfdna-cancer-detection"
    job_queue = "cfdna-processing-queue"
    
    parameters = {
        "inputConfigPath": config_path,
        "outputPath": output_path,
        "modelType": model_type,
        "experimentId": experiment_id
    }
    
    # Container overrides for the specific job
    container_overrides = {
        "command": [
            "python", "-m", "cfdna.train",
            f"s3://{config_path}",
            model_type,
            f"s3://{output_path}",
            "--seed", "42"
        ],
        "environment": [
            {"name": "EXPERIMENT_ID", "value": experiment_id},
            {"name": "JOB_NAME", "value": job_name}
        ]
    }
    
    # Build AWS CLI command
    cmd_parts = [
        "aws batch submit-job",
        f"--job-name {job_name}",
        f"--job-queue {job_queue}",
        f"--job-definition {job_def}",
        f"--parameters '{json.dumps(parameters)}'",
        f"--container-overrides '{json.dumps(container_overrides)}'"
    ]
    
    return " \\\n  ".join(cmd_parts)


def submit_job_with_boto3(job_name: str, config_path: str, model_type: str,
                         output_path: str, experiment_id: str = "default") -> Dict[str, Any]:
    """Submit job using boto3.
    
    Args:
        job_name: Name for the batch job
        config_path: S3 path to config file
        model_type: Type of model to train
        output_path: S3 path for outputs
        experiment_id: Experiment identifier
        
    Returns:
        Batch job response
    """
    import boto3
    
    batch_client = boto3.client("batch")
    
    job_definition = "cfdna-cancer-detection"
    job_queue = "cfdna-processing-queue"
    
    parameters = {
        "inputConfigPath": config_path,
        "outputPath": output_path,
        "modelType": model_type,
        "experimentId": experiment_id
    }
    
    container_overrides = {
        "command": [
            "python", "-m", "cfdna.train",
            f"s3://{config_path}",
            model_type,
            f"s3://{output_path}",
            "--seed", "42"
        ],
        "environment": [
            {"name": "EXPERIMENT_ID", "value": experiment_id},
            {"name": "JOB_NAME", "value": job_name}
        ]
    }
    
    response = batch_client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        parameters=parameters,
        containerOverrides=container_overrides,
        tags={
            "Project": "cfDNA-Cancer-Detection",
            "ExperimentId": experiment_id,
            "ModelType": model_type
        }
    )
    
    return response


def main() -> None:
    """Main function for submitting batch jobs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Submit cfDNA job to AWS Batch")
    parser.add_argument("--job-name", default="cfdna-training-job",
                       help="Name for the batch job")
    parser.add_argument("--config-path", 
                       default="your-cfdna-config-bucket/configs/synthetic_config.yaml",
                       help="S3 path to configuration file")
    parser.add_argument("--model-type", default="mlp",
                       choices=["logistic_l1", "logistic_l2", "random_forest", "mlp"],
                       help="Type of model to train")
    parser.add_argument("--output-path",
                       default="your-cfdna-output-bucket/results/",
                       help="S3 path for output artifacts")
    parser.add_argument("--experiment-id", default="default",
                       help="Experiment identifier")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show command without executing")
    
    args = parser.parse_args()
    
    print("AWS Batch Job Submission for cfDNA Cancer Detection")
    print("=" * 55)
    print(f"Job Name: {args.job_name}")
    print(f"Config Path: {args.config_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Output Path: {args.output_path}")
    print(f"Experiment ID: {args.experiment_id}")
    print()
    
    # Check AWS credentials
    has_credentials = check_aws_credentials()
    
    if not has_credentials or args.dry_run:
        print("AWS credentials not available or dry-run mode.")
        print("Here is the AWS CLI command that would be executed:")
        print()
        
        cli_command = generate_batch_command(
            args.job_name, args.config_path, args.model_type,
            args.output_path, args.experiment_id
        )
        
        print(cli_command)
        print()
        print("To run this command:")
        print("1. Configure AWS credentials: aws configure")
        print("2. Ensure the job definition and queue exist")
        print("3. Upload your config file to S3")
        print("4. Execute the command above")
        
    else:
        print("AWS credentials found. Submitting job...")
        
        try:
            response = submit_job_with_boto3(
                args.job_name, args.config_path, args.model_type,
                args.output_path, args.experiment_id
            )
            
            print("Job submitted successfully!")
            print(f"Job ID: {response['jobId']}")
            print(f"Job ARN: {response['jobArn']}")
            print()
            print("Monitor job status with:")
            print(f"aws batch describe-jobs --jobs {response['jobId']}")
            
        except ImportError:
            print("boto3 not installed. Install with: pip install boto3")
            sys.exit(1)
            
        except Exception as e:
            print(f"Error submitting job: {e}")
            print()
            print("Falling back to CLI command:")
            
            cli_command = generate_batch_command(
                args.job_name, args.config_path, args.model_type,
                args.output_path, args.experiment_id
            )
            print(cli_command)
            sys.exit(1)


if __name__ == "__main__":
    main()