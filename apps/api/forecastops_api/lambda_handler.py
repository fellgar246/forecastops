"""API Gateway entrypoint for the cloud control plane.

The function serves the same HTTP routes as the local process. Credentials
come from the execution role.
"""

from mangum import Mangum

from forecastops_api.main import app

handler = Mangum(app, lifespan="off")
