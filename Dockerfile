FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY .env.example .

# Environment Defaults
ENV PORT=8001
ENV WORKERS=4

# Expose ports
EXPOSE 8001 8502

# Startup script using python wrapper
CMD ["python", "scripts/entrypoint.py"]
