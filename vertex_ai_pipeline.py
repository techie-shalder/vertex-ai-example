import json
import os
import kfp
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint

from kfp import compiler
from kfp.dsl import (
    Artifact,
    Input,
    Dataset,
    Model,
)

# Set the path to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"

class MyEndpoint(Artifact):
    _schema_title_ = 'MyEndpoint'
    _schema_version_ = 'v1'

    def __init__(self, name: str):
        super(MyEndpoint, self).__init__()
        self.name = name

# Define custom component functions

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery", "google-cloud-aiplatform"])
def create_bigquery_dataset() -> Dataset:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    dataset_id = "austin_311"
    # Initialize BigQuery client
    service_account_info = json.load(open('./key.json'))
    print(service_account_info)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    client = bigquery.Client(credentials=credentials)
    
    # Create dataset
    dataset = bigquery.Dataset(client.dataset(dataset_id))
    dataset = client.create_dataset(dataset, exists_ok=True)
    
    return dataset

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery", "google-cloud-aiplatform"])
def export_bigquery_dataset(dataset: Input[Dataset]) -> Artifact:
    from google.cloud import bigquery
    
    # Initialize BigQuery client
    # Create credentials from the service account file
    credential = credentials.Credentials.from_service_account_file("key.json")

    client = bigquery.Client(credentials=credential)
    
    # Export dataset
    job_config = bigquery.ExtractJobConfig(destination_format="CSV")
    job = client.extract_table(
        dataset.uri, "311_service_requests", destination_uris=["gs://bucket-shining-granite-414702-01/exported_dataset.csv"], job_config=job_config
    )
    job.result()
    
    # Return exported artifact
    return Artifact("gs://bucket-shining-granite-414702-01/exported_dataset.csv")

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def train_xgboost_model(dataset: Input[Artifact]) -> Model:
    import xgboost as xgb
    import pandas as pd
    
    # Load dataset
    data = pd.read_csv(dataset.uri)
    X, y = data.drop(columns=["target"]), data["target"]
    
    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X, y)
    
    # Save model
    model.save_model("/tmp/model.bst")
    
    # Return trained model artifact
    return Model("/tmp/model.bst")

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def create_endpoint()-> MyEndpoint:
    from google.cloud import aiplatform
    
    # Initialize Vertex AI client
    # Create credentials from the service account file
    credential = credentials.Credentials.from_service_account_file("key.json")
    client = aiplatform.gapic.EndpointServiceClient(credentials=credential)
    
    # Create endpoint
    endpoint = client.create_endpoint(display_name="my_endpoint")

    return Endpoint(name=endpoint.name)

@kfp.dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_to_endpoint(model: Input[Model], endpoint: Input[MyEndpoint]):
    from google.cloud import aiplatform
    
    # Initialize Vertex AI client
     # Create credentials from the service account file
    credential = credentials.Credentials.from_service_account_file("key.json")
    
    client = aiplatform.gapic.EndpointServiceClient(credentials=credential)
    
    # Deploy model to endpoint
    deployed_model = client.deploy_model(endpoint.name, model=model)
    
    return MyEndpoint(name=deployed_model.endpoint)

# Define pipeline
@kfp.dsl.pipeline(
    name="Vertex AI Pipeline",
    description="A pipeline to create and deploy an XGBoost model on Vertex AI.",
    pipeline_root="gs://bucket-shining-granite-414702-01/vertex_ai_pipeline/pipeline_root/",
)
def vertex_ai_pipeline():
    create_dataset_task = create_bigquery_dataset()
    export_dataset_task = export_bigquery_dataset(dataset=create_dataset_task.output).after(create_bigquery_dataset)
    train_model_task = train_xgboost_model(dataset=export_dataset_task.output)
    create_endpoint_task = create_endpoint()
    deploy_model_task = deploy_model_to_endpoint(model=train_model_task.output, endpoint=create_endpoint_task.output)

# Compile and run the pipeline
if __name__ == "__main__":
    project_id = "shining-granite-414702"
    location = "us-central1"
    compiler.Compiler().compile(
        pipeline_func=vertex_ai_pipeline,
        package_path="vertex_ai_pipeline.json",
    )
    # Configure the pipeline
job = aiplatform.PipelineJob(
    display_name="vertex_ai_pipeline",
    template_path="vertex_ai_pipeline.json",
    pipeline_root="gs://bucket-shining-granite-414702-01/vertex_ai_pipeline/pipeline_root/",
    enable_caching=False,
    location=location,
    project= project_id,
)

# Run the job
job.run(service_account="service-account-1@shining-granite-414702.iam.gserviceaccount.com")
