FROM python:3.11-slim

WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8000

# Command to run the application (example for Flask)
CMD ["gunicorn", "-w", "4", "app:app"]
