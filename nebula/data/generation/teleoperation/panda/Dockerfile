FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker_requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY server.py /app/server.py

WORKDIR /app
COPY protos /app/protos
COPY mp_pb2.py /app/mp_pb2.py
COPY mp_pb2_grpc.py /app/mp_pb2_grpc.py
EXPOSE 50051
ENTRYPOINT ["python3", "server.py"]
