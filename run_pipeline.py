from zenml.client import Client
from zenml.pipelines.base_pipeline import BasePipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

class TrainingPipeline(BasePipeline):
    def connect(self):
        ingest = ingest_data()
        clean = clean_data(ingest)
        train = train_model(clean)
        evaluate = evaluation(train)
        return evaluate

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )