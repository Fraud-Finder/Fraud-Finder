# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: default command
CMD ["python3"]
