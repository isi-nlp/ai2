FROM python:3.7-stretch
WORKDIR /source

# Install dependencies.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy remaining code.
COPY . .

RUN mkdir /results

CMD ["/bin/bash"]