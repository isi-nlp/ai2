FROM python:3.7-stretch
WORKDIR /source

# Install dependencies.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy remaining code.
COPY config/ ./config/
COPY model_cache/ ./model_cache/
COPY eval.py joint_large_8.ckpt model.py run_model.sh sub_cycic.py ./

RUN mkdir /results

# Run code.
CMD ["/usr/bin/env", "bash"]
