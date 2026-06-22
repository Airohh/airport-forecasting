FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY data/processed/ data/processed/
COPY reports/ reports/
COPY models/ models/

RUN pip install --no-cache-dir -e ".[serving]" && \
    pip install --no-cache-dir uvicorn

EXPOSE 8000

CMD ["uvicorn", "airport_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
