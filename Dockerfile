# I used python version 3.11, so I put it here too
FROM python:3.11-slim

# Installing dependencies for the operating system
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the file with dependencies
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying application files to the container
COPY . .

# Command to run uvicorn when container starts
CMD ["uvicorn", "classifier_fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
