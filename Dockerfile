FROM python:3.6

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
ADD weights/best.h5 weights/best.h5
ADD src/model.py src/model.py
ADD app.py app.py

# Install required libraries
RUN pip install -r requirements.txt

# Run it once to trigger resnet download
RUN python app.py

EXPOSE 8008

# Start the server
CMD ["python", "app.py", "serve"]