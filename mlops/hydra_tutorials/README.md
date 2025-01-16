With the `MLProject` defined, we can now run the project using `mlflow run` command and can override the default
parameters when needed:

```bash
mlflow run [path_to_project] -P hydra_options="main.experiment_nam=exp2"
```

We can also reset the `n_estimators` parameter in the `random_forest` subsection of the `random_forest_pipeline` section:

```bash
mlflow run [path_to_project] -P hydra_options="random_forest_pipeline.random_forest.n_estimators=100"
```

We can also override multiple parameters at once:

```bash
mlflow run [path_to_project] -P hydra_options="main.experiment_name=exp2, main.project_name=project2"
```

