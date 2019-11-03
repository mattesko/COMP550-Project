FROM python:3.7-slim-buster

COPY NB_submission.py /usr/src/

COPY naive_bayes_w_articles.sav /usr/src/

COPY vectorizer.pk /usr/src/

RUN pip install --upgrade pip

RUN pip install sklearn

RUN pip install pandas

CMD [ "python3", "/usr/src/NB_submission.py" ]
