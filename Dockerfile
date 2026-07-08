# Start from a minimal Python environment — small image, faster builds/downloads
FROM python:3.11-slim

# Set the working folder inside the container (everything below happens relative to this)
WORKDIR /app

# Copy ONLY requirements first (not the whole project yet).
# This is a deliberate ordering trick: Docker caches each step, so if your
# code changes but requirements.txt doesn't, Docker skips re-installing
# everything and reuses the cached layer — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual app code and the trained model files
COPY src/ ./src/
COPY models/ ./models/

# Document which port the app listens on (informational — doesn't publish it by itself)
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
