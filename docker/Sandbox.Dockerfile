FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.5
COPY docker/sandbox_runner.py /opt/sandbox_runner.py
ENTRYPOINT ["python", "-I", "/opt/sandbox_runner.py"]
