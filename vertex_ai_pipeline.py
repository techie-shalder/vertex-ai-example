import kfp
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint

from kfp import compiler
from kfp.dsl import (Artifact, Dataset, Input, Output)


# Set the path to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
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
def export_bigquery_dataset():
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

def deploy_model_to_endpoint(model: Input[Artifact]) -> Output[Artifact]:
    # Initialize AIPlatformClient
    ai_client = aiplatform.AIPlatformClient()

    # Deploy the trained model to the endpoint
    model_path = model.path  # Assuming the model artifact contains the path to the model
    deployed_model = ai_client.create_endpoint(deployed_model=model_path)

    # Return the deployed model (if needed)
    return deployed_model

# Define your KFP pipeline
@kfp.dsl.pipeline(
        name="vertex-ai-pipeline",
        description="My Pipeline Description",
        pipeline_root="gs://bucket-shining-granite-414702-01/vertex_ai_pipeline/pipeline_root/"
)
def vertex_ai_pipeline():
    # Create BigQuery dataset
    bigquery_dataset_task = create_bigquery_dataset()
    
    # Export the BigQuery dataset
    export_dataset_task = export_bigquery_dataset().after(bigquery_dataset_task)
    print(export_dataset_task.outputs.keys())

    # # Train an XGBoost model
    # train_xgboost_model_task = train_xgboost_model(export_dataset_task.outputs['exported_data'])
    
    # # Create an endpoint
    # create_endpoint_task = create_endpoint()
    
    # # Deploy the model to the endpoint
    # deploy_model_task = deploy_model_to_endpoint(
    #     model=train_xgboost_model_task.outputs['model'],
    #     endpoint=create_endpoint_task.outputs['endpoint'])
   
# Compile the KFP pipeline
compiler.Compiler().compile(
    pipeline_func=vertex_ai_pipeline,
    package_path="vertex_ai_pipeline.json"
)

# Define your GCP project and KFP location
project_id = "shining-granite-414702"
location = "us-central1"

# # Initialize the AI Platform client
# ai_client = aiplatform.AIPlatformClient(project_id=project_id, region=location)

# # Upload and run the pipeline on KFP
# ai_client.create_run_from_job_spec("vertex_ai_pipeline.json")
# Configure the pipeline
job = aiplatform.PipelineJob(
    display_name="vertex_ai_pipeline",
    template_path="vertex_ai_pipeline.json",
    pipeline_root="gs://bucket-shining-granite-414702-01/vertex_ai_pipeline/pipeline_root/",
    enable_caching=False,
    location="us-central1",
    project= project_id,
)

# Run the job
job.run()
