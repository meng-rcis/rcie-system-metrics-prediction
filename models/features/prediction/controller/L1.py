import concurrent.futures
import pandas as pd
import config.os as os_config

from interface import IL1
from config.control import L1_MODELS, SETUP_L1_CONFIG, PREDICTION_L1_CONFIG


# NOTE: Purpose of the L1 is to let the user to define the base models and its configurations in a single place
class L1(IL1):
    def __init__(
        self,
        model_ids: list[str],
        is_parallel_processing: bool = False,
    ):
        self.models = self.InitiateModels(model_ids)
        self.is_parallel_processing = is_parallel_processing

    # NOTE: A function to prepare all models
    def InitiateModels(self, model_ids: list[str]) -> list[dict]:
        models = []
        for model_id in model_ids:
            model = {
                "id": model_id,
                "instance": L1_MODELS.get(model_id, None),
                "setup_config": SETUP_L1_CONFIG.get(model_id, {}),
                "prediction_config": PREDICTION_L1_CONFIG.get(model_id, {}),
            }
            if model["instance"] is None:
                raise Exception(f"Model {model_id} in L1 is not supported")
            print("model_id", model_id)
            print("model setup config", model["setup_config"])
            print("model prediction config", model["prediction_config"], "\n")
            models.append(model)
        return models

    # NOTE: A function to execute the training process of all models
    def TrainModels(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int = 0,
        end_index: int = None,
        steps: int = 1,
    ):
        if self.is_parallel_processing:
            self.__parallel_model_train(
                dataset=dataset,
                feature=feature,
                start_index=start_index,
                end_index=end_index,
                steps=steps,
            )
        else:
            self.__sequential_model_train(
                dataset=dataset,
                feature=feature,
                start_index=start_index,
                end_index=end_index,
                steps=steps,
            )

    # NOTE: A function to execute the prediction process of all models
    def Predict(self, steps: int) -> pd.DataFrame:
        predictions = pd.DataFrame()

        for model in self.models:
            prediction = model["instance"].Predict(
                config={**model["prediction_config"], "steps": steps}
            )
            predictions[model["id"]] = prediction

        return predictions

    def __parallel_model_train(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int = 0,
        end_index: int = None,
        steps: int = 1,
    ):
        # Prepare the parameters for each model
        for model in self.models:
            model["instance"].PrepareParameters(
                dataset=dataset,
                feature=feature,
                start_index=start_index,
                end_index=end_index,
            )

        # Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os_config.MAXIMUM_NUMBER_OF_PROCESS
        ) as executor:
            futures = [
                executor.submit(
                    model["instance"].TrainModel,
                    {
                        **model["setup_config"],
                        "steps": steps,
                    },
                )
                for model in self.models
            ]
            # Retrieve results in the order of submission
            trained_models = [future.result() for future in futures]

        # Sequentially saving the trained models
        for model, trained_model in zip(self.models, trained_models):
            model["instance"].SaveModel(trained_model)

    def __sequential_model_train(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int = 0,
        end_index: int = None,
        steps: int = 1,
    ):
        for model in self.models:
            model["instance"].PrepareParameters(
                dataset=dataset,
                feature=feature,
                start_index=start_index,
                end_index=end_index,
            )
            model["instance"].TrainModel(
                config={**model["setup_config"], "steps": steps}
            )
