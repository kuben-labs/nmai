
FROM python:3.13-slim

# Install curl and the Azure CLI
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    git \
    lsb-release \
    gnupg \
    build-essential \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

RUN uv pip install --system --no-cache .

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080 8083

CMD ["/app/entrypoint.sh"]
