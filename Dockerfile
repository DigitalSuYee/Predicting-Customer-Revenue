FROM python:3.11.6-slim


RUN pip install pipenv


WORKDIR /app                                                                


COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --deploy --system


COPY ["*.py", "model_C=0.01.bin", "./"]


EXPOSE 9696


ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "revenue:app"]