import mlflow
import hydra
import os

# automatically reads the configuration file in config.yaml
@hydra.main(config_name="config")
def go(config):
    # config is a dictionary
    print(config)

    # set the environment variables for wandb to group the runs
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # load the mlflow configuration
    project_name = config['main']['project_name']
    mlflow.run(
        uri="test_data",
        parameters={
            "input_artifact": config['data']['train_data'],
            "output_artifact": "raw_data.csv",
            # ...
        },
    )

    mlflow.run(
        uri="train_random_forest",
        parameters={
            "input_artifact": "raw_data.csv:latest",
            "n_estimators": config['random_forest_pipeline']['random_forest']['n_estimators'],
            "max_depth": config['random_forest_pipeline']['random_forest']['max_depth'],
            # ...
        }
    )

if __name__ == "__main__":
    go()
