# Use the official Python image as the base image
FROM python:3.12.1-slim-bullseye
 
# Set the working directory to /app
WORKDIR /app
 
# Copy the requirements file into the container at /app
COPY requirements.txt .
RUN apt update \
&& apt install -y pkg-config \
&& apt install -y default-libmysqlclient-dev build-essential \
&& apt clean
 
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
 
# Expose port 8000 for the application
EXPOSE 8000