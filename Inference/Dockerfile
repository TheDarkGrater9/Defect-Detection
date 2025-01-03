# Use the official NVIDIA PyTorch base image for GPU support
FROM nvcr.io/nvidia/pytorch:23.03-py3

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY inference.py .

# Expose the entry point for the script
ENTRYPOINT ["python", "inference.py"]
