FROM python:3.8
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# COPY . /app
# WORKDIR /app
# RUN pip install -e .

# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY
# RUN dvc pull data/01_raw/hateful_memes
