# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Buat folder upload jika belum ada
RUN mkdir -p static/uploads

# Jalankan Flask
CMD ["python", "Apps/app.py"]
