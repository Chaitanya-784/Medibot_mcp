# Step 1: Use an official Python image from Docker Hub
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the contents of your local project directory (the 'model' directory) to the container at /app
COPY . /app
# Step 4: Install any dependencies you need, based on requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 6: Expose the port the app will run on
EXPOSE 8086

# Step 7: Define the command to run your app using Gunicorn + Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:$PORT", "mcp_server:app"]
