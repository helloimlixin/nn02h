# Introduction to YAML

## What is YAML?

YAML stands for "YAML Ain't Markup Language" or "Yet Another Markup Language". It is a human-readable data serialization
language that is commonly used for configuration files and data exchange. YAML is often used in conjunction with
programming languages like Python, Ruby, and Perl. It is a cross-language format that is designed to store data structures
in a human-readable format. Similar to JSON and XML, YAML is a way to represent data in a structured format but has fewer
boilerplate characters and is easier to read.

In YAML, just like in Python, indentation is used to define the structure of the data. This makes YAML files easy to read
and write for humans. YAML files typically end with the `.yaml` or `.yml` file extension. Note that tabs are not allowed
for indentation in YAML files, and spaces must be used instead.

## Basic Syntax

YAML uses a simple and intuitive syntax to represent data. Here are some basic rules and conventions for writing YAML:

1. **Indentation**: YAML uses indentation to define the structure of the data. Indentation is done using spaces (not tabs)
   and is typically two or four spaces per level. The number of spaces per level should be consistent throughout the file.
2. **Key-Value Pairs**: Data in YAML is represented as key-value pairs. The key and value are separated by a colon (`:`).
   The key is followed by a colon and a space, and the value can be a string, number, boolean, list, or another object.
3. **Lists**: Lists in YAML are represented using a hyphen (`-`) followed by a space. Each item in the list is preceded
   by a hyphen and a space, and the items can be of different types.
4. **Comments**: Comments in YAML start with a hash (`#`) symbol and continue to the end of the line. Comments can be used
   to provide additional information or context about the data.
5. **Quoting**: Strings in YAML can be enclosed in single quotes (`'`) or double quotes (`"`). Single quotes are used to
   preserve the literal value of the string, while double quotes allow for escape sequences and special characters.
6. **Block Scalar**: YAML supports block scalar style for multiline strings. Block scalar strings are enclosed in
   `|` or `>`, and preserve newlines and leading/trailing whitespace.
7. **Anchors and Aliases**: YAML allows you to define anchors (`&`) and aliases (`*`) to reuse data within the same file.
   An anchor is defined using `&` followed by a name, and an alias is defined using `*` followed by the name of the anchor.
8. **References**: YAML supports references (`*`) to anchor points within the same file. References are used to point to
   previously defined anchors and can be used to avoid duplication of data.
9. **Directives**: YAML directives are used to specify instructions for the YAML parser. Directives start with `%` and are
   followed by a directive name and parameters.
10. **Tags**: YAML tags are used to specify the data type of a value. Tags start with `!` and are followed by a tag name.
    Tags can be used to specify the type of a value or to provide additional information about the data.
11. **Flow Style**: YAML supports flow style for representing complex data structures like maps and lists in a single line.
    Flow style uses brackets (`{}` and `[]`) and commas to separate elements.
12. **Document Separators**: YAML allows multiple documents to be stored in a single file. Documents are separated by three
    hyphens (`---`) followed by a newline.
13. **Special Characters**: YAML supports special characters like `&`, `*`, `!`, `|`, `>`, `%`, `#`, `@`, `^`, and `:`. These
    characters have special meanings in YAML and should be used with caution.
14. **Reserved Words**: YAML has reserved words like `yes`, `no`, `true`, `false`, `null`, `on`, `off`, `~`, `n/a`, and
    `NULL` that have special meanings and should be used as-is.
15. **Escaping**: YAML supports escaping of special characters using backslashes (`\`). This allows you to include special
    characters in strings without them being interpreted as control characters.
16. **Implicit Types**: YAML has implicit typing rules that allow it to automatically convert values to specific data types.
    For example, numbers, booleans, and null values are automatically converted to their corresponding data types.

## Lists

Lists in Python:

```python
my_list = [1, 2, 'yoho', 4, 5]
```

Equivalent list in YAML:

```yaml
my_list:
  - 1
  - 2
  - 'yoho'
  - 4
  - 5
```

Nested lists in Python:

```python
nested_list = [1, 2, 3, 'bang', [7, 'ooo', 9]]
```

Equivalent nested list in YAML:

```yaml
nested_list:
  - 1
  - 2
  - 3
  - 'bang'
  - - 7
    - 'ooo'
    - 9
```

## Dictionaries

Dictionaries in Python:

```python
my_dict = {
    'name': 'Alice',
    'age': 30,
    'address': {
        'street': '123 Main St',
        'city': 'Wonderland',
        'zip': '12345'
    },
    'phone': ['123-456-7890', '987-654-3210']
}
```

Equivalent dictionary in YAML:

```yaml
my_dict:
  name: Alice
  age: 30
  address:
    street: 123 Main St
    city: Wonderland
    zip: 12345
  phone:
    - 123-456-7890
    - 987-654-3210
```

Dictionaries with lists

```python
dict_with_list = {
    'name': 'Bob',
    'age': 25,
    'interests': ['music', 'movies', 'sports']
}
```

Equivalent dictionary with list in YAML:

```yaml
dict_with_list:
  name: Bob
  age: 25
  interests:
    - music
    - movies
    - sports
```

## The `conda.yml` File and MLProject

Recall the `conda.yml` file defined:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - requests=2.25.1
  - pip=21.0.1
  - pip:
      - wandb==0.10.30
```

can be parsed using the following Python code:

```python
import yaml

with open('conda.yml', 'r') as file:
    conda_config = yaml.safe_load(file)

print(conda_config)
```

which prints out the following dictionary:

```python
{
    'name': 'download_data',
    'channels': ['conda-forge', 'defaults'],
    'dependencies': [
        'requests=2.25.1',
        'pip=21.0.1',
        {'pip': ['wandb==0.10.30']}
    ]
}
```

The `mlproject` file defined:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      file_url:
        description: URL of the file to download
        type: uri
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str
      artifact_type:
        description: Type of the artifact to create
        type: str
        default: raw_data
      artifact_description:
        description: Description for the artifact
        type: str

    command: >-
      python download_data.py --file_url {file_url} \
                              --artifact_name {artifact_name} \
                              --artifact_type {artifact_type} \
                              --artifact_description {artifact_description}
```

can be parsed using the following Python code:

```python
import yaml

with open('mlproject', 'r') as file:
    mlproject_config = yaml.safe_load(file)

print(mlproject_config)
```

which prints out the following dictionary:

```python
{
    'name': 'download_data',
    'conda_env': 'conda.yml',
    'entry_points': {
        'main': {
            'parameters': {
                'file_url': {
                    'description': 'URL of the file to download',
                    'type': 'uri'
                },
                'artifact_name': {
                    'description': 'Name for the W&B artifact that will be created',
                    'type': 'str'
                },
                'artifact_type': {
                    'description': 'Type of the artifact to create',
                    'type': 'str',
                    'default': 'raw_data'
                },
                'artifact_description': {
                    'description': 'Description for the artifact',
                    'type': 'str'
                }
            },
            'command': 'python download_data.py --file_url {file_url} --artifact_name {artifact_name} --artifact_type {artifact_type} --artifact_description {artifact_description}'
        }
    }
}
```

