FROM mambaorg/micromamba:1.2.0
COPY environment.yml .	
RUN micromamba install -n base --file environment.yml --yes


USER root
RUN chown mambauser:mambauser /
RUN pip install yahooquery
#RUN python -m install yahooquery
WORKDIR "/"
CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root"]
