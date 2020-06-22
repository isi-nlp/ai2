FROM python:3.7-stretch
WORKDIR /source

# Install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy model
COPY model_cache/ ./model_cache/
COPY cycic_2020_06_17_s42.ckpt ./

# Copy remaining code
COPY config/ ./config/
COPY eval.py model.py run_model.sh sub_cycic.py ./

# Make output directory
RUN mkdir /results

# Run code
CMD ["/usr/bin/env", "bash"]