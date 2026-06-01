# Use Python 3.13 with Slim base image
FROM python:3.13-slim

# Set working directory
WORKDIR /usr/src/app

# Copy the application code into the image
COPY ./conf /conf

# Install libgl1 for OpenCV and wget for downloading the model
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# Download the machine learning model
RUN wget https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx -O /conf/model/model-weights-5a6b1be1fa.onnx

# Upgrade pip and install the required Python packages
RUN pip install --upgrade pip && pip install numpy==2.4.6 opencv-python==4.13.0.92 appdaemon==4.5.13 onnxruntime==1.26.0 pyyaml

# Start the AppDaemon service
CMD ["appdaemon", "-c", "/conf"]
