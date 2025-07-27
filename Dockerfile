# Base image (lightweight and fast)
FROM python:3.10-slim

# Disable internet access inside container (runtime)
ENV no_proxy="*"

# Create working directory
WORKDIR /app

# Copy everything (analyze_collections.py, collections, etc.)
COPY . /app

# Install required packages (offline-friendly)
RUN pip install --no-cache-dir \
    PyPDF2 \
    scikit-learn \
    nltk

# Pre-download NLTK data during build so runtime works offline
RUN python -m nltk.downloader punkt stopwords

# Set entrypoint to script
ENTRYPOINT ["python", "src/analyze_collections.py"]
