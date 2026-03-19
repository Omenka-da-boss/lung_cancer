FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY src/app/ ./app
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t lung_cancer:latest .