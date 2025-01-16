FROM python:3.10-slim

WORKDIR /usr/src

# Copy only the requirements file, to leverage Docker’s caching mechanism
COPY requirements.txt .

# Install dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# (Optional) Specify the command to run your application
# For example, if you have an app.py, you can do:
# CMD ["python", "app.py"]
CMD = ["bash"]