"""Shared API error body."""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Every error response uses this shape."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class ApiError(Exception):
    """Domain failure returned as :class:`ErrorBody`."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = ErrorBody(code=code, message=message, details=details or {})
        super().__init__(message)


def install_error_handlers(application: FastAPI) -> None:
    """Replace framework error payloads with the shared body."""

    @application.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorBody(
            code="validation_error",
            message="Request validation failed.",
            details={"errors": jsonable_encoder(exc.errors())},
        )
        return JSONResponse(status_code=422, content=body.model_dump())
