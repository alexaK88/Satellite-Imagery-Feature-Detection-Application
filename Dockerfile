FROM ubuntu:latest
LABEL authors="aleksandrak"

ENTRYPOINT ["top", "-b"]