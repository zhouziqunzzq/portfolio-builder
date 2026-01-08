from __future__ import annotations


class BrokerApiError(Exception):
    """Base class for broker adapter errors."""


class TemporaryUnavailable(BrokerApiError):
    """Retryable, transient broker/API failure."""


class RateLimited(BrokerApiError):
    def __init__(
        self, message: str = "rate limited", *, retry_after_seconds: float | None = None
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class OrderNotFoundYet(BrokerApiError):
    """Retryable: order not visible yet due to eventual consistency."""


class InvalidOrder(BrokerApiError):
    """Non-retryable: order request is invalid for this broker."""


class OrderRejected(BrokerApiError):
    """Non-retryable: broker rejected the order."""


class NotTradable(BrokerApiError):
    """Non-retryable: instrument cannot be traded."""


class AuthError(BrokerApiError):
    """Fatal: authentication/authorization error."""
