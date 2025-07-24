# Base image with PyTorch + CUDA support for GPU 
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["python", "your_backend_file.py"]
# Replace with your actual entry file name if different
