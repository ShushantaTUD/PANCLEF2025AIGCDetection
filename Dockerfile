# Use an official Python base image
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the default command to run the script
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python3", "deberta_submission.py", "--input-file", "$inputDataset/dataset.jsonl", "--output-dir", "$outputDir"]

