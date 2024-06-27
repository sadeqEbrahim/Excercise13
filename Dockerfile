# docker build --tag python-docker .  		# Build the image
# docker run -d -p 8080:8080 python-docker 	# Run the container
# docker ps                                 # Check if the container is running
# docker logs xxxxxxx                       # Logs in case of error

FROM python:3.8-slim-buster
WORKDIR /python-docker
COPY . /python-docker
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python3", "app.py"]
