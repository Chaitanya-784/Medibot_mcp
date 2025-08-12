# Use an official Python slim image for a smaller footprint.
FROM python:3.10-slim

# Set environment variables for better Python behavior in containers.
# 1. PYTHONUNBUFFERED: Ensures logs are sent directly to the container's output.
# 2. PYTHONDONTWRITEBYTECODE: Prevents Python from creating .pyc files.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory for all subsequent commands.
WORKDIR /app

# --- Step 1: Install Dependencies ---
# Copy only the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if your requirements.txt file changes.
COPY requirements.txt .

# Upgrade pip and install dependencies from requirements.txt in a single RUN command.
# This reduces the number of layers in the final image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Step 2: Pre-download NLTK data ---
# This is a critical optimization. By downloading the NLTK data during the
# image build, you prevent slow downloads when a new container starts,
# which was a likely cause of your previous deployment timeouts.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# --- Step 3: Copy Application Code ---
# Now that dependencies are installed and cached, copy the rest of your
# application code (like mcp_server.py, model.h5, etc.) into the container.
COPY . .

# --- Step 4: Add a Non-Root User for Security ---
# Running containers as a non-root user is a major security best practice.
# We create a dedicated user and group to run the application.
RUN addgroup --system appgroup && adduser --system --group appuser

# Switch to the newly created non-root user.
USER appuser

# --- Step 5: Expose Port and Run Application ---
# Expose the port your application will listen on. This is metadata for Docker.
EXPOSE 8080

# Define the command to run your app using Gunicorn, a production-ready server.
# Using a single worker (-w 1) is important because your app uses an in-memory
# session dictionary, which cannot be shared across multiple processes.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8080", "mcp_server:app"]