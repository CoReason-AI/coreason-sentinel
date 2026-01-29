# Requirements

Coreason Sentinel requires Python 3.12+ and the following dependencies:

*   **FastAPI**: Web framework for building APIs.
*   **Uvicorn**: ASGI server for running the FastAPI application.
*   **Redis**: In-memory data store for Circuit Breaker state and metrics.
*   **Pydantic**: Data validation and settings management.
*   **NumPy & SciPy**: Statistical computations for drift detection.
*   **HTTPX**: Async HTTP client for external service integration.
*   **AnyIO**: Asynchronous I/O support.
*   **Loguru**: Logging library.

## Installation

Install via pip:

```bash
pip install coreason-sentinel
```

Or using Poetry:

```bash
poetry add coreason-sentinel
```
