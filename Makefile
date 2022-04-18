export POETRY=poetry run
export PWD=`pwd`
export DATA_DIR=data
export DOCKER_IMAGE_NAME=padim
export ARCH=wide_resnet50_2


init: # Setup pre-commit
	poetry install
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint: # Lint all files in this repository
	${POETRY} pre-commit run --all-files --show-diff-on-failure

run: # Run locally
	${POETRY} python src/main.py --arch ${ARCH}

build_docker: # Build docker image
	docker build -t ${DOCKER_IMAGE_NAME} .

run_docker:
	docker run --rm -v ${PWD}/${DATA_DIR}:/opt/data ${DOCKER_IMAGE_NAME} --arch ${ARCH}
