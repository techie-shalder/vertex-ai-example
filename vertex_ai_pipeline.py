from google.cloud import aiplatform, storage
import kfp
from kfp import components
from kfp import dsl

# Define component functions
@components.create_component_from_func
def create_bigquery_dataset(project_id: str, dataset_id: str):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset.location = "US"
    dataset = client.create_dataset(dataset, exists_ok=True)

@components.create_component_from_func
def export_dataset_to_gcs(project_id: str, dataset_id: str, gcs_path: str):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.job.ExtractJobConfig()
    job_config.destination_format = bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
    dataset_ref = client.dataset(dataset_id, project=project_id)
    job = client.extract_table(dataset_ref.table("YOUR_TABLE_NAME"), gcs_path, job_config=job_config)
    job.result()

@components.create_component_from_func
def train_xgboost_model(training_data_path: str, model_output_path: str):
    import xgboost as xgb
    import pandas as pd

    # Load data
    data = pd.read_csv(training_data_path)
    X = data.drop(columns=['target'])
    y = data['target']

    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Save model
    model.save_model(model_output_path)

@components.create_component_from_func
def create_endpoint(project_id: str, region: str, endpoint_name: str):
    from google.cloud import aiplatform

    # Initialize the AI Platform client
    aiplatform.init(project=project_id, location=region)

    # Create an endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    return endpoint.name

@components.create_component_from_func
def deploy_model_to_endpoint(project_id: str, region: str, model_path: str, endpoint_id: str):
    from google.cloud import aiplatform

    # Initialize the AI Platform client
    aiplatform.init(project=project_id, location=region)

    # Get the model resource
    model = aiplatform.Model.upload(
        display_name="MyModel",
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/xgboost-cpu.0-82:latest",
    )

    # Deploy the model to the endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)
    deployed_model = endpoint.deploy(model)

    return deployed_model.name

# # Function to upload file to GCS
# def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)

#     blob.upload_from_filename(source_file_name)

# Define the pipeline
@dsl.pipeline(
    name="MyPipeline",
    description="A pipeline to create a BigQuery dataset, export dataset, train XGBoost model, and deploy it."
)
def my_pipeline(
    project_id: str,
    dataset_id: str,
    gcs_path: str,
    training_data_path: str,
    model_output_path: str,
    endpoint_name: str,
    region: str,
):
    create_bigquery_dataset_op = create_bigquery_dataset(project_id, dataset_id)
    export_dataset_to_gcs_op = export_dataset_to_gcs(project_id, dataset_id, gcs_path)
    train_xgboost_model_op = train_xgboost_model(training_data_path, model_output_path)
    create_endpoint_op = create_endpoint(project_id, region, endpoint_name)
    deploy_model_to_endpoint_op = deploy_model_to_endpoint(project_id, region, model_output_path, create_endpoint_op.output)

    create_bigquery_dataset_op.after(export_dataset_to_gcs_op)
    train_xgboost_model_op.after(export_dataset_to_gcs_op)
    deploy_model_to_endpoint_op.after(train_xgboost_model_op)
    # upload_to_gcs("your-bucket-name", "vertex_ai_pipeline.json", "vertex_ai_pipeline.json")



# Compile the pipeline
pipeline_func = my_pipeline
pipeline_filename = "vertex_ai_pipeline.json"
import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func, pipeline_filename)

# # Submit the pipeline to Vertex AI Pipelines
# pipeline_job = aiplatform.PipelineJob(
#     display_name="Vertex AI Pipeline",
#     template_path=pipeline_file,
#     enable_caching=False  # Disable caching to ensure the pipeline runs fresh every time
# )
# pipeline_job.run(sync=True)

# Execute the pipeline
pipeline_args = {
    'project_id': 'shining-granite-414702',
    'dataset_id': 'dataset01',
    'gcs_path': 'gs://bucket-shining-granite-414702-01/dataset_export/',
    'training_data_path': 'gs://bucket-shining-granite-414702-01/training_data.csv',
    'model_output_path': 'gs://bucket-shining-granite-414702-01/model/',
    'endpoint_name': 'endpoint01',
    'region': 'us-central1'
}

# client = kfp.Client()
# exp = client.create_experiment(name='my_experiment')

# run_name = 'MyPipeline_run'
# run_result = client.run_pipeline(exp.id, run_name, pipeline_filename, params=pipeline_args)
