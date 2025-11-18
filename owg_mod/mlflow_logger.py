import mlflow
from typing import Dict, Any, Optional
from datetime import datetime

class MLflowLogger:
    def __init__(self, experiment_name: str = "OWG_Experiments", tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        mlflow.log_dict(dictionary, artifact_file)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def end_run(self):
        if self.run is not None:
            mlflow.end_run()
            self.run = None
