"""
Azure ML Pipelines — end-to-end MLOps on Azure.

Provides:
- Pipeline definition with preprocessing, training, evaluation, registration steps
- Responsible AI dashboard integration
- Model registry and versioning
- Real-time and batch inference endpoints
- MLflow experiment tracking
- Automated hyperparameter tuning (sweep jobs)
- Data asset management
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    AmlCompute,
    CommandComponent,
    Data,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    SweepJob,
)
from azure.ai.ml.sweep import Choice, Uniform
from azure.identity import DefaultAzureCredential

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRunResult:
    """Result of an Azure ML pipeline job."""

    job_name: str
    status: str
    studio_url: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "Completed"

    @property
    def failed(self) -> bool:
        return self.status == "Failed"


@dataclass
class EndpointInfo:
    """Information about a deployed Azure ML online endpoint."""

    endpoint_name: str
    scoring_uri: str
    deployment_name: str
    instance_type: str
    status: str


class AzureMLOrchestrator:
    """
    Production Azure ML pipeline orchestrator.

    Builds, submits, and monitors Azure ML pipelines including
    data preprocessing, training, responsible AI evaluation,
    model registration, and endpoint deployment.

    Example:
        ml = AzureMLOrchestrator()

        # Submit a training pipeline
        result = ml.submit_training_pipeline(
            experiment_name="churn-prediction",
            training_data="azureml:churn-dataset:1",
            compute_cluster="gpu-cluster",
        )

        # Wait for completion
        if result.succeeded:
            endpoint = ml.deploy_model(
                model_name="churn-model",
                model_version="1",
            )
            print(f"Scoring URI: {endpoint.scoring_uri}")
    """

    def __init__(self) -> None:
        raw = load_config()
        ml_cfg = raw.get("azure_ml", {})

        self._workspace = ml_cfg.get("workspace_name", "")
        self._resource_group = ml_cfg.get("resource_group", "")
        self._subscription_id = ml_cfg.get("subscription_id", "")
        self._ml_cfg = ml_cfg

        self._ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group,
            workspace_name=self._workspace,
        )

        mlflow_tracking_uri = ml_cfg.get("mlflow", {}).get("tracking_uri", "")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        logger.info(
            "AzureMLOrchestrator initialised",
            workspace=self._workspace,
            resource_group=self._resource_group,
        )

    def submit_training_pipeline(
        self,
        experiment_name: str,
        training_data: str,
        compute_cluster: str | None = None,
        environment_name: str = "azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        hyperparameters: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PipelineRunResult:
        """
        Submit a training pipeline with preprocessing, training, and evaluation.

        Args:
            experiment_name: MLflow experiment name.
            training_data: Azure ML data asset path (e.g. "azureml:dataset:version").
            compute_cluster: Compute cluster name (overrides config).
            environment_name: Azure ML environment for training.
            hyperparameters: Model hyperparameters dict.
            tags: Job tags for filtering.

        Returns:
            PipelineRunResult with job status and Studio URL.
        """
        cluster = compute_cluster or self._ml_cfg.get("compute", {}).get("training_cluster", "cpu-cluster")
        hparams = hyperparameters or {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}

        preprocess_component = self._build_preprocessing_component(environment_name)
        train_component = self._build_training_component(environment_name, hparams)
        evaluate_component = self._build_evaluation_component(environment_name)

        @pipeline(name=f"{experiment_name}-pipeline", description="End-to-end training pipeline")
        def training_pipeline(raw_data: Input) -> dict:
            preprocess_step = preprocess_component(raw_data=raw_data)
            preprocess_step.compute = cluster

            train_step = train_component(
                train_data=preprocess_step.outputs.train_data,
                val_data=preprocess_step.outputs.val_data,
            )
            train_step.compute = cluster

            evaluate_step = evaluate_component(
                model=train_step.outputs.model,
                test_data=preprocess_step.outputs.test_data,
            )
            evaluate_step.compute = cluster

            return {"evaluation_report": evaluate_step.outputs.report}

        pipeline_job = training_pipeline(
            raw_data=Input(type=AssetTypes.URI_FILE, path=training_data)
        )
        pipeline_job.experiment_name = experiment_name
        pipeline_job.tags = tags or {}

        try:
            submitted = self._ml_client.jobs.create_or_update(pipeline_job)
        except Exception as exc:
            logger.error("Pipeline submission failed", error=str(exc))
            raise

        logger.info(
            "Pipeline job submitted",
            job_name=submitted.name,
            experiment=experiment_name,
            studio_url=submitted.studio_url,
        )

        return PipelineRunResult(
            job_name=submitted.name,
            status=submitted.status,
            studio_url=submitted.studio_url or "",
        )

    def wait_for_job(
        self,
        job_name: str,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 7200,
    ) -> PipelineRunResult:
        """
        Wait for an Azure ML pipeline job to complete.

        Args:
            job_name: The submitted job name.
            poll_interval_seconds: Status check interval.
            timeout_seconds: Max wait time (default 2 hours).

        Returns:
            PipelineRunResult with final status.
        """
        terminal = {"Completed", "Failed", "Canceled"}
        elapsed = 0

        while elapsed < timeout_seconds:
            job = self._ml_client.jobs.get(job_name)
            status = job.status
            logger.debug("Pipeline job status", job_name=job_name, status=status)

            if status in terminal:
                return PipelineRunResult(
                    job_name=job_name,
                    status=status,
                    studio_url=job.studio_url or "",
                )

            time.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

        raise TimeoutError(f"Pipeline job {job_name} did not complete within {timeout_seconds}s")

    def run_sweep_job(
        self,
        experiment_name: str,
        training_data: str,
        compute_cluster: str | None = None,
        max_total_trials: int = 20,
        max_concurrent_trials: int = 4,
    ) -> str:
        """
        Run hyperparameter sweep using Azure ML's built-in tuning.

        Searches over: n_estimators, max_depth, learning_rate.

        Args:
            experiment_name: Experiment name.
            training_data: Data asset URI.
            compute_cluster: Compute cluster name.
            max_total_trials: Total HPO trials.
            max_concurrent_trials: Concurrent trial count.

        Returns:
            The sweep job name.
        """
        cluster = compute_cluster or self._ml_cfg.get("compute", {}).get("training_cluster", "cpu-cluster")

        command_job = command(
            code="./scripts",
            command="python train.py --training_data ${{inputs.training_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --learning_rate ${{inputs.learning_rate}}",
            inputs={
                "training_data": Input(type=AssetTypes.URI_FILE, path=training_data),
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            compute=cluster,
            environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
            experiment_name=experiment_name,
        )

        sweep = command_job.sweep(
            sampling_algorithm="bayesian",
            primary_metric="val_accuracy",
            goal="maximize",
        )

        sweep.search_space = {
            "n_estimators": Choice([50, 100, 150, 200]),
            "max_depth": Choice([3, 5, 6, 8, 10]),
            "learning_rate": Uniform(min_value=0.01, max_value=0.3),
        }

        sweep.limits = {
            "max_total_trials": max_total_trials,
            "max_concurrent_trials": max_concurrent_trials,
            "timeout": 7200,
        }

        sweep.early_termination = {
            "type": "bandit",
            "evaluation_interval": 2,
            "slack_factor": 0.1,
        }

        submitted = self._ml_client.jobs.create_or_update(sweep)
        logger.info("Sweep job submitted", job_name=submitted.name, max_trials=max_total_trials)
        return submitted.name

    def register_model(
        self,
        model_path: str,
        model_name: str,
        model_version: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> Model:
        """
        Register a model in the Azure ML Model Registry.

        Args:
            model_path: Local path or Azure ML job output URI to model artifacts.
            model_name: Registry name for the model.
            model_version: Version string (auto-incremented if None).
            description: Model description.
            tags: Metadata tags.

        Returns:
            The registered Model object.
        """
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            description=description,
            tags=tags or {},
        )

        registered = self._ml_client.models.create_or_update(model)
        logger.info(
            "Model registered",
            name=registered.name,
            version=registered.version,
        )
        return registered

    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        endpoint_name: str | None = None,
        instance_type: str | None = None,
        instance_count: int = 1,
        traffic_percentage: int = 100,
    ) -> EndpointInfo:
        """
        Deploy a registered model to an Azure ML managed online endpoint.

        Args:
            model_name: Registered model name.
            model_version: Model version to deploy.
            endpoint_name: Endpoint name (auto-generated if not provided).
            instance_type: VM size (overrides config).
            instance_count: Number of instances.
            traffic_percentage: Traffic percentage routed to this deployment.

        Returns:
            EndpointInfo with scoring URI and deployment details.
        """
        ep_name = endpoint_name or f"{model_name}-endpoint-{int(time.time())}"
        inst_type = instance_type or self._ml_cfg.get("compute", {}).get("inference_vm_size", "Standard_DS3_v2")

        # Create or update endpoint
        endpoint = ManagedOnlineEndpoint(name=ep_name, auth_mode="key")
        self._ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        # Create deployment
        deployment_name = f"{model_name}-v{model_version}"
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=ep_name,
            model=f"azureml:{model_name}:{model_version}",
            instance_type=inst_type,
            instance_count=instance_count,
        )

        self._ml_client.online_deployments.begin_create_or_update(deployment).result()

        # Route traffic
        endpoint.traffic = {deployment_name: traffic_percentage}
        self._ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        # Get scoring URI
        ep_info = self._ml_client.online_endpoints.get(ep_name)
        scoring_uri = ep_info.scoring_uri or ""

        logger.info(
            "Model deployed",
            endpoint=ep_name,
            deployment=deployment_name,
            scoring_uri=scoring_uri,
        )

        return EndpointInfo(
            endpoint_name=ep_name,
            scoring_uri=scoring_uri,
            deployment_name=deployment_name,
            instance_type=inst_type,
            status="Succeeded",
        )

    def create_compute_cluster(
        self,
        cluster_name: str,
        vm_size: str = "Standard_DS3_v2",
        min_instances: int = 0,
        max_instances: int = 4,
    ) -> AmlCompute:
        """Create or update an Azure ML compute cluster."""
        cluster = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size=vm_size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=120,
        )
        result = self._ml_client.compute.begin_create_or_update(cluster).result()
        logger.info("Compute cluster created", name=cluster_name, vm_size=vm_size, max_instances=max_instances)
        return result

    # ------------------------------------------------------------------
    # Private component builders
    # ------------------------------------------------------------------

    def _build_preprocessing_component(self, environment: str) -> CommandComponent:
        return CommandComponent(
            name="data_preprocessing",
            display_name="Data Preprocessing",
            description="Clean, split, and feature-engineer raw training data",
            inputs={"raw_data": Input(type=AssetTypes.URI_FILE)},
            outputs={
                "train_data": Output(type=AssetTypes.URI_FOLDER),
                "val_data": Output(type=AssetTypes.URI_FOLDER),
                "test_data": Output(type=AssetTypes.URI_FOLDER),
            },
            code="./scripts",
            command="python preprocess.py --raw_data ${{inputs.raw_data}} --train_data ${{outputs.train_data}} --val_data ${{outputs.val_data}} --test_data ${{outputs.test_data}}",
            environment=environment,
        )

    def _build_training_component(self, environment: str, hparams: dict[str, Any]) -> CommandComponent:
        return CommandComponent(
            name="model_training",
            display_name="Model Training",
            description="Train a scikit-learn model with MLflow tracking",
            inputs={
                "train_data": Input(type=AssetTypes.URI_FOLDER),
                "val_data": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"model": Output(type=AssetTypes.MLFLOW_MODEL)},
            code="./scripts",
            command=(
                "python train.py "
                "--train_data ${{inputs.train_data}} "
                "--val_data ${{inputs.val_data}} "
                f"--n_estimators {hparams.get('n_estimators', 100)} "
                f"--max_depth {hparams.get('max_depth', 6)} "
                f"--learning_rate {hparams.get('learning_rate', 0.1)} "
                "--output_model ${{outputs.model}}"
            ),
            environment=environment,
        )

    def _build_evaluation_component(self, environment: str) -> CommandComponent:
        return CommandComponent(
            name="model_evaluation",
            display_name="Model Evaluation",
            description="Evaluate model on test data and generate responsible AI report",
            inputs={
                "model": Input(type=AssetTypes.MLFLOW_MODEL),
                "test_data": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"report": Output(type=AssetTypes.URI_FOLDER)},
            code="./scripts",
            command="python evaluate.py --model ${{inputs.model}} --test_data ${{inputs.test_data}} --report ${{outputs.report}}",
            environment=environment,
        )
