FROM pytorch/pytorch 
WORKDIR /source

# Install dependencies.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy remaining code.
COPY . .

# Run code.
CMD ["python", "-u"]
