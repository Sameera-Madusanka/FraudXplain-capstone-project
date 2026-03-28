FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY webapp/backend/requirements.txt /app/webapp/backend/requirements.txt
RUN pip install --no-cache-dir -r webapp/backend/requirements.txt

# Copy project files
COPY config.py /app/
COPY data_loader_bank.py /app/
COPY models/ /app/models/
COPY federated_learning/ /app/federated_learning/
COPY explainability/ /app/explainability/
COPY utils/ /app/utils/
COPY results/ /app/results/
COPY data/protected_attributes.json /app/data/
COPY webapp/backend/ /app/webapp/backend/

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "webapp/backend/app.py"]
