import kfp
from google.cloud import aiplatform
from kfp.v2 import compiler
from kfp.v2.dsl import (Dataset, Input, Model, Model)
from kfp.v2.google.client import AIPlatformClient

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery", "google-cloud-aiplatform"])
def create_bigquery_dataset():
    from google.cloud import bigquery

    # Initialize BigQuery client
    bq_client = bigquery.Client()

    # Define your BigQuery dataset ID
    dataset_id = "shining-granite-414702.austin_311"

    # Create the BigQuery dataset
    dataset = bq_client.create_dataset(dataset_id, exists_ok=True)

    return dataset

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery", "google-cloud-aiplatform"])
def export_bigquery_dataset(dataset: Input[Dataset]):
    from google.cloud import bigquery

    # Initialize BigQuery client
    bq_client = bigquery.Client()

    # Define your BigQuery dataset ID
    dataset_id = "shining-granite-414702.austin_311"

    # Define your GCS destination URI for export
    gcs_uri = "gs://bucket-shining-granite-414702-01/exported_dataset.csv"

    # Export the BigQuery dataset to GCS
    job = bq_client.extract_table(dataset_id, "311_service_requests", gcs_uri)

    return job

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def train_xgboost_model(dataset: Input[Dataset]):
    # Define your training script and parameters
    training_script = "gs://bucket-shining-granite-414702-01/train.py"
    job_dir = "gs://bucket-shining-granite-414702-01/job_dir"
    region = "us-central1"
    project = "shining-granite-414702"

    # Train the XGBoost model
    aiplatform.ModelTrainingJob(
        display_name="xgboost-training-job",
        script_path=training_script,
        container_uri="gcr.io/cloud-aiplatform/training/xgboost-cpu.1-4:latest",
        model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-4:latest",
        requirements=["google-cloud-bigquery"],
        args=[
            "--dataset-id", dataset.id,
            "--job-dir", job_dir,
            "--region", region,
            "--project", project
        ],
    )

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def create_endpoint():
    # Define your Endpoint display name
    endpoint_display_name = "xgboost-endpoint"

    # Create the Endpoint
    endpoint = aiplatform.Endpoint.create(endpoint_display_name)

    return endpoint

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_to_endpoint(model: Input[Model], endpoint: Input[Endpoint]):
    # Deploy the trained model to the endpoint
    endpoint.deploy(model=model, machine_type="n1-standard-4")

# Define your KFP pipeline
@kfp.dsl.pipeline(name="vertex-ai-pipeline")
def vertex_ai_pipeline():
    bigquery_dataset_task = create_bigquery_dataset()
    export_dataset_task = export_bigquery_dataset(bigquery_dataset_task.output)
    print(export_dataset_task)
    train_xgboost_model_task = train_xgboost_model(bigquery_dataset_task.output)
    endpoint_creation_task = create_endpoint()
    model_deployment_task = deploy_model_to_endpoint(train_xgboost_model_task.output, endpoint_creation_task.output)
    print(model_deployment_task)
# Compile the KFP pipeline
compiler.Compiler().compile(
    pipeline_func=vertex_ai_pipeline,
    package_path="vertex_ai_pipeline.json"
)

# Define your GCP project and KFP location
project_id = "shining-granite-414702"
location = "us-central1"

# Initialize the AI Platform client
ai_client = AIPlatformClient(project_id=project_id, region=location)

# Upload and run the pipeline on KFP
ai_client.create_run_from_job_spec("vertex_ai_pipeline.json")
