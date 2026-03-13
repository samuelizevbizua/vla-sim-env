# Use Python 3.11 slim to guarantee compatibility with pre-built Linux wheels
# This completely bypasses the C++ compilation errors you saw on Windows
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version to ensure smooth installations
RUN pip install --upgrade pip

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the dependencies (pip will seamlessly grab the Linux wheels here!)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Set the default command to run when the container starts
# (Update 'main.py' to whatever your actual starting script is named)
CMD ["python", "main.py"]