# Stage 1: Build the React Application
FROM node:18 AS frontend-build
WORKDIR /app/frontend
# Copy package files and install dependencies
COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN npm install
# Copy the rest of the frontend source code and build it
COPY webapp/frontend/ ./
RUN npm run build

# Stage 2: Build the Python Flask Server
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies required for TensorFlow and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY webapp/backend/requirements.txt /app/webapp/backend/requirements.txt
RUN pip install --no-cache-dir -r webapp/backend/requirements.txt

# Copy all the primary application files
COPY config.py /app/
COPY data_loader_bank.py /app/
COPY models/ /app/models/
COPY federated_learning/ /app/federated_learning/
COPY explainability/ /app/explainability/
COPY utils/ /app/utils/
COPY data/ /app/data/
COPY results/ /app/results/
COPY webapp/backend/ /app/webapp/backend/

# Crucial Step: Copy the compiled React application from Stage 1 into the Flask directory
# This allows Flask to perfectly serve both the API and the User Interface from a single running container!
COPY --from=frontend-build /app/frontend/dist /app/webapp/backend/static

# Suppress annoying TensorFlow warnings in the cloud
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1

# Expose the default Flask port internally just in case (though Railway overrides it dynamically)
EXPOSE 5000

# Finally, execute the main entrypoint
CMD ["python", "webapp/backend/app.py"]
