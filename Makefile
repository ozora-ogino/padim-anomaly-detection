export POETRY=poetry run
export PWD=`pwd`
export DOCKER_IMAGE_NAME=padim

# Arguments for python scripts.
export data_path=data
export save_path=mvtec_result
export arch=wide_resnet50_2
export archive_path=archive
export threshold=0.5


init: # Setup pre-commit
	poetry install
	pre-commit install --hook-type pre-commit --hook-type pre-push

create_dataset:
	${POETRY} python src/split_dataset.py --archive_path ${archive_path} --data_path ${data_path}

lint: # Lint all files in this repository
	${POETRY} pre-commit run --all-files --show-diff-on-failure

run: # Run locally
	${POETRY} python src/main.py --arch ${arch} --save_path ${save_path} --threshold ${threshold}

build_docker: # Build docker image
	docker build -t ${DOCKER_IMAGE_NAME} .

run_docker:
	docker run --rm -v ${PWD}/${data_path}:/opt/data -v ${PWD}/${save_path}:/opt/mvtec_result \
	${DOCKER_IMAGE_NAME} --arch ${arch} --threshold ${threshold}
