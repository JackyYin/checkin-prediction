FROM python:3.5
EXPOSE 80

# 安裝系統library
RUN apt-get update -y; \
    apt-get install -y --no-install-recommends apt-utils libgdal-dev;

# 安裝工具
RUN apt-get install -y --no-install-recommends netcat vim sudo

# 安裝相依套件
RUN mkdir /code
COPY ./requirements.txt /code
WORKDIR /code
RUN pip install -r requirements.txt;

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /code

ENTRYPOINT ["bash", "-c"]
CMD ["python main.py"]

