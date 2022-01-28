FROM python:3.8-alpine as builder
RUN apk update
RUN apk add curl
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
COPY . .
RUN $HOME/.poetry/bin/poetry build

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN apt-get update
RUN apt-get install -y git
RUN pip install dgl-cu110 ogb
COPY --from=builder dist/mlap-0.1.0-py3-none-any.whl .
RUN pip install --no-deps mlap-0.1.0-py3-none-any.whl
COPY runner.py .
ENV DGLBACKEND=pytorch
ENV MLAP_ROOT="/workspace"
ARG githash
ENV GITHASH=$githash
ENTRYPOINT ["python", "-u", "runner.py"]
CMD ["--train"]
