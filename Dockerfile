FROM python:3

COPY requirements.txt ta-lib-0.4.0-src.tar.gz /
COPY ./src /src
COPY ./SP-data /SP-data

RUN ls -l && tar zxf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure && make && make install && ldconfig -v
RUN python -m pip install -r requirements.txt

WORKDIR /src

ENTRYPOINT ["python", "-u", "main.py"]
