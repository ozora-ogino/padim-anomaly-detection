FROM python:3.8

WORKDIR /opt
COPY pyproject.toml poetry.lock ./
RUN pip install -U pip && pip install poetry && \
	poetry config virtualenvs.create false && \
	poetry install --no-dev

COPY ./src/ src
ENTRYPOINT ["python", "src/main.py"]

