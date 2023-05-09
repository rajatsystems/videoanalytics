FROM python:3.11

RUN mkdir /videoanalytics
WORKDIR /videoanalytics
ADD requirements.txt /videoanalytics/
RUN python3 -m pip install -r requirements.txt
ADD . /videoanalytics/

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
