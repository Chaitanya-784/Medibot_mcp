# Step 1: Use an official Python image from Docker Hub.
# The `slim` version is a good choice for smaller, more efficient containers.
FROM python:3.10-slim

# Step 2: Set the working directory inside the container.
# All subsequent commands will be run from this directory.
WORKDIR /app

# Step 3: Copy all files from your local project directory into the container's /app directory.
# This includes your Python code, requirements.txt, and any other necessary files.
COPY . /app

# Step 4: Install Python dependencies. Upgrading pip first is a good practice.
# The `--no-cache-dir` flag helps keep the container image size small.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port the app will run on. This is for documentation purposes.
# The actual port is determined by the `PORT` environment variable at runtime.
EXPOSE 8080

# Step 6: Define the command to run your app.
# The `-b 0.0.0.0:8080` part tells Gunicorn to bind to the port 8080 on all network interfaces.
# Gunicorn automatically picks up the PORT environment variable if it's not explicitly defined
# here, or you can explicitly pass it. However, since your Python code is already
# configured to read the PORT variable, specifying 8080 here is a common and reliable pattern.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8080", "mcp_server:app"]
