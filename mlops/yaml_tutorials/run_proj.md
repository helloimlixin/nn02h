# Running the Project

Running an MLflow project is accomplished by using the `mlflow run` command. This command runs a project from a local
directory, a Git URI, or a databricks workspace. The command requires a URI to the project, and it can also take a list
of parameters that are passed to the project. The parameters are passed as key-value pairs, and are specified using the
`-P` flag: `-P [parameter_name]=[paramter_value]`, each parameter is specified by a separate `-P` flag.
# Run default script from a local folder
```bash
mlflow run ./my_project -P file_url=https://... -P artifact_name=my_data.csv
```

# Run from a local folder but with a different entry point
```bash
mlflow run ./my_project -e other_script -P parameter_one=15
```

You can also run a project from a Git URI by substituting the local path with a GitHub URI. This is done by specifying
the Git URI as the project URI.

# Run from a Git URI
```bash
mlflow run git@github.com/[myusername]/[myrepo].git -P file_url=https://... -P artifact_name=my_data.csv
```

You can also specify a release tag using the `-v` flag (which is often preferred to run the released version instead
of the dev version) or a branch using the `-b` flag.

# Run from a Git URI with a release tag
```bash
mlflow run git@github.com/[myusername]/[myrepo].git -v 1.2.3
```

