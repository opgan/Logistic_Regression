install:
		pip install --upgrade pip &&\
			pip install -r requirements.txt

test:
		python -m pytest -vv --cov=hello test_hello.py 

format:
		black *.py lib/*.py

lint:
		pylint --disable=R,C *.py lib/*.py

refactor: format lint

all: install lint test format