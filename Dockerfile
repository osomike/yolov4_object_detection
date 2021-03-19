FROM python:3.8-slim

# Copy requirements and binary files
COPY ./binaries/yolov4.h5 ./binaries/yolov4.h5
COPY ./app/requirements.txt ./app/requirements.txt

# Install requirements
RUN apt-get update
RUN	apt-get install -y gcc vim
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN python -m pip install --upgrade pip
RUN pip install -r ./app/requirements.txt

# Copy main application
COPY ./app/tools.py ./app/tools.py
COPY ./app/main.py ./app/main.py

# set working directory
WORKDIR /app

# Only the last command will be executed
CMD ["uvicorn", "main:object_detection_server", "--host", "0.0.0.0", "--port", "8001"]
# docker build . -t yolov4_docker
# docker run -it yolov4_docker bash
# docker run -p 8000:8001 yolov4_docker