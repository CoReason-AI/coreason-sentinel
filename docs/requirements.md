# Requirements

## System Requirements

- **Python:** >= 3.12, < 3.15
- **Redis:** >= 5.0.0 (Server)

## Python Dependencies

The core dependencies for `coreason-sentinel` are:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `fastapi` | ^0.128.0 | Web framework for the service API |
| `uvicorn` | ^0.40.0 | ASGI server for production |
| `redis` | ^5.0.0 | State persistence and sliding window metrics |
| `pydantic` | ^2.12.5 | Data validation and settings management |
| `numpy` | ^2.4.1 | Numerical operations for drift detection |
| `scipy` | ^1.16.3 | Scientific computing (KL Divergence, Cosine Similarity) |
| `loguru` | ^0.7.2 | Enhanced logging |
| `httpx` | ^0.28.1 | Async HTTP client for external services |
| `aiofiles` | * | Async file operations |
| `anyio` | * | Asynchronous compatibility layer |

## Development Dependencies

For running tests and contributing:

- `pytest`
- `pytest-cov`
- `pytest-asyncio`
- `ruff` (Linting and Formatting)
- `pre-commit`
- `mkdocs` (Documentation)
