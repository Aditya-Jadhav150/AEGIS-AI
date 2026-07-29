# Build a slim Python environment optimized for heavy machine learning
FROM python:3.11-slim

# Install system-level dependencies for computer vision (OpenCV/Pillow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Required directories for Flask + uploads
RUN mkdir -p static/uploads instance \
    && chmod -R 777 static/uploads instance

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# ✅ FIXED: Install compatible PyTorch version (LOCKED)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Remove conflicting torch entries from requirements
RUN sed -i '/^torch==/d' requirements.txt && \
    sed -i '/^torchvision==/d' requirements.txt && \
    sed -i '/^torchaudio==/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install --no-cache-dir gunicorn

# Hugging Face port
EXPOSE 7860

# Run Flask app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers=2", "--timeout=120", "app:app"]