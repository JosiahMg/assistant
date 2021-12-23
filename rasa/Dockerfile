FROM python:3.7-slim

RUN python -m pip install --upgrade pip
RUN python -m pip install rasa==3.0.3

WORKDIR /app
COPY . .

RUN rasa train

# set the user to run, don't run as root
USER 1001

# set entrypoint for interactive shells
ENTRYPOINT ["rasa"]

CMD ["run", "--enable-api", "--port", "8080"]

# build : docker build -t mh/rasa3-1.0 .
# run: docker run -it -p 8080:8080 mh/rasa3-1.0