"""Packaged Lambda entrypoint.

The deploy command copies this module to ``handler.py`` at the root of the
deployment package. API Gateway calls ``handler.handler``.
"""

from forecastops_api.lambda_handler import handler

__all__ = ["handler"]
