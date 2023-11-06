build:
	bash ./bash/build.sh -$(FLAG)

help:
	@make build FLAG=h

conda_env:
	conda activate syntax

conda_env_install:
	pip3 install -r requirements.txt
