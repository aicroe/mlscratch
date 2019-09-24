## Requirements

* Python 3.6 or higher

## Setup your environment

* Clone the project
* Create a dedicated virtual environment
```bash
cd mlscratch
python3 -m venv python3
source python3/bin/activate
```

## Install for development
```bash
pip install -e ".[dev]"
```

## Run the tests
```bash
python setup.py test
```

## Run linter
```bash
pylint src
```
The above will output a friendly colorized report, if needed it can be avoided by appending the option: `--output-format=text`
