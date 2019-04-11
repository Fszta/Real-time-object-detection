FROM ubuntu:16.04

FROM python:3

ADD requirements.txt /

ADD MobileNetSSD_deploy.caffemodel /

ADD MobileNetSSD_deploy.prototxt.txt /

ADD object_detection.py /

RUN pip install -r requirements.txt

CMD [ "python", "./object_detection.py"]
