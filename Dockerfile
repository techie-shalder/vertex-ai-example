# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable for Google Cloud SDK installation
ENV CLOUD_SDK_VERSION=356.0.0

# Install Google Cloud SDK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        python3-dev \
        gcc \
        apt-transport-https \
        lsb-release && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk=${CLOUD_SDK_VERSION}-0 && \
    apt-get remove -y --purge \
        curl \
        python3-dev \
        gcc \
        apt-transport-https && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    gcloud --version

EXPOSE 8080

CMD ["python", "pipeline.py"]
