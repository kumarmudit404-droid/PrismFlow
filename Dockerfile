# PrismFlow diagnostic scoring service -- Hugging Face Spaces, CPU only.
#
# TORCH IS CPU-ONLY AND DELIBERATELY SO. The default PyPI wheel carries the
# CUDA runtime: roughly 2 GB of libraries for a GPU a Space does not have.
# The +cpu wheel from PyTorch's own index is a fraction of that and runs the
# same code, so it is installed first, from its own index, before anything
# else can resolve a CUDA build as a transitive dependency.
#
# THE SERVICE DOES NOT RUN AS ROOT. A process that reads files chosen by
# strangers has no business owning the filesystem it runs on, so a plain
# user owns the app directory and the container drops to it before the
# entrypoint. Spaces expects uid 1000, which is what `app` gets here.
#
# NOTHING SECRET IS BAKED IN. There is no key, token or credential in this
# image or in the build context it needs. ALLOWED_ORIGIN is supplied at run
# time, and with it unset the service allows no browser origin at all.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app

COPY --chown=app:app serve/requirements.txt ./serve/requirements.txt

# CPU wheel first, from PyTorch's index, so no CUDA build is ever resolved.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 \
 && pip install --no-cache-dir -r serve/requirements.txt

COPY --chown=app:app prismflow ./prismflow
COPY --chown=app:app serve ./serve

USER app

ENV PYTHONPATH=/home/app
EXPOSE 7860

# One worker: the scoring path holds the whole upload in memory, and more
# workers would multiply the peak rather than the throughput.
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
