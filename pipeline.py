import kfp
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    ComponentStore,
)
from kfp.v2.google import big_query, xgboost, ai_platform

# Define component store
components = ComponentStore(
    url="https://raw.githubusercontent.com/kubeflow/pipelines/master/components/gcp/"
)

@dsl.pipeline(
    name="Vertex AI Pipeline",
    description="A pipeline to demonstrate Vertex AI capabilities"
)
def vertex_ai_pipeline(
    project_id: str,
    dataset_name: str,
    table_name: str,
    model_display_name: str,
    endpoint_display_name: str,
    region: str = "us-central1",
):
    # Step 1: Create a BigQuery dataset
    bq_create_dataset = big_query.create_dataset_op(
        project_id=project_id, dataset_id=dataset_name
    )

    # Step 2: Export the dataset
    bq_export_op = big_query.export_big_query_to_gcs_op(
        project_id=project_id,
        dataset_id=dataset_name,
        table_id=table_name,
        gcs_uri=f"gs://{bucket_name}/{dataset_name}/",
    )

    # Step 3: Train an XGBoost model
    xgboost_train_op = xgboost.train_op(
        training_data=bq_export_op.outputs["output_gcs_uri"],
        project=project_id,
        display_name=model_display_name,
    )

    # Step 4: Create an Endpoint resource
    create_endpoint_op = ai_platform.EndpointCreateOp(
        display_name=endpoint_display_name, project=project_id
    )

    # Step 5: Deploy the Model resource to the Endpoint resource
    deploy_model_op = ai_platform.ModelDeployOp(
        model=xgboost_train_op.outputs["model"],
        endpoint=create_endpoint_op.outputs["endpoint"],
    )

    # Compile the pipeline
    pipeline = dsl.Pipeline(
        name="Vertex AI Pipeline",
        description="A pipeline to demonstrate Vertex AI capabilities",
        pipeline_root="gs://your-bucket/path/to/pipeline_root",
        components=[bq_create_dataset, bq_export_op, xgboost_train_op, create_endpoint_op, deploy_model_op],
    )

    return pipeline

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=vertex_ai_pipeline,
    package_path="vertex_ai_pipeline.json",
)

# Execute the pipeline using Vertex AI Pipelines
api_client = kfp.Client()
run_name = "vertex-ai-pipeline-run"
experiment_name = "Vertex AI Pipeline Experiment"
run_result = api_client.create_run_from_pipeline_package(
    pipeline_file="vertex_ai_pipeline.json",
    run_name=run_name,
    experiment_name=experiment_name,
    params={
        "project_id": "your-project-id",
        "dataset_name": "your-dataset-name",
        "table_name": "your-table-name",
        "model_display_name": "your-model-display-name",
        "endpoint_display_name": "your-endpoint-display-name",
    },
)
