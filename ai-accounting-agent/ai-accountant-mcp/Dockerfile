
FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

RUN uv pip install --system --no-cache .

EXPOSE 8083

CMD ["uv", "run", "python", "src/ai_accountant_mcp/server.py"]
