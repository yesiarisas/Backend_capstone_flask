FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Buka port
EXPOSE 8080

# Jalankan aplikasi Flask via Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
