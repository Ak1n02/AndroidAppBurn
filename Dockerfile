# Use an official Python image with CUDA if needed
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git curl && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download the model file
RUN python download_model.py

# Expose port for Render
ENV PORT 5000
EXPOSE 5000

# Run the app
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]
