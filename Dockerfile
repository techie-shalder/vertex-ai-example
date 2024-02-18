# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the pipeline script and any additional files
COPY . /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the pipeline script when the container launches
CMD ["python", "vertex_ai_pipeline.py"]
