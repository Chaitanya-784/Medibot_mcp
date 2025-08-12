# Step 1: Base image (compatible with TensorFlow & dependencies)
FROM python:3.10-slim

# Step 2: Environment settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Step 3: Set work directory
WORKDIR /app

# Step 4: Install system deps for python-magic, libgl (needed by Pillow), and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy project files to container
COPY . /app

# Step 6: Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 7: Pre-download NLTK data to avoid download on startup
RUN python -m nltk.downloader punkt wordnet

# Step 8: Ensure model and data files exist
# (If they’re in a separate dir, adjust COPY step accordingly)

# Step 9: Expose the port (match Cloud Run PORT)
EXPOSE 8080

# Step 10: Start server with Gunicorn+Uvicorn — read PORT dynamically
CMD exec gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT mcp_server:app
