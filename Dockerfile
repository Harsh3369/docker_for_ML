# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /physcian_conversion_model_docker

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# # Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
