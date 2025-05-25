
FROM ubuntu:20.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y


RUN pip3 install opencv_python onnxruntime requests  multi-model-server \
                sagemaker-inference retrying

COPY dockered-entrypoint.py /usr/local/bin/dockered-entrypoint.py
RUN chmod +x /usr/local/bin/dockered-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the custom service file to handle incoming data and inference requests
COPY scripts/dockered-inference.py /home/model-server/dockered-inference.py
COPY scripts/passport_extractor.py /home/model-server/passport_extractor.py
COPY scripts/passport_detector.py /home/model-server/passport_detector.py
COPY scripts/ocr_inference.py /home/model-server/ocr_inference.py


RUN mkdir -p /home/raw-data/
RUN chmod +rwx /home/raw-data/

# Entrypoint script for the docker image
ENTRYPOINT ["python3", "/usr/local/bin/dockered-entrypoint.py"]


# Command to be passed to the entrypoint
CMD ["serve"]
