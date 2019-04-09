FROM python:3

WORKDIR /app

COPY . ./

RUN pip install djangorestframework
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install sklearn
RUN pip install matplotlib

CMD [ "python", "manage.py runserver" ]