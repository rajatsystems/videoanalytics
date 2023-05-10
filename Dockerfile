FROM python:3.11

RUN mkdir /videoanalytics
WORKDIR /videoanalytics

ADD requirements.txt /videoanalytics/
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN python3 -m pip install -r requirements.txt

ADD . /videoanalytics/

# Add the entrypoint command to start the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]


