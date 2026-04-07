"""
Structured logging for Azure AI Platform Engineering.
Integrates with Azure Application Insights via OpenTelemetry.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    service_name: str = "azure-ai-platform",
    appinsights_connection_string: str | None = None,
) -> None:
    """
    Configure structlog with optional Azure Application Insights export.

    Args:
        level: Log level string.
        json_output: Emit JSON (for Log Analytics). False = coloured console.
        service_name: Service name stamped on every record.
        appinsights_connection_string: If provided, configure OTLP export to App Insights.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, level.upper()))

    for noisy in ("azure", "httpx", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if appinsights_connection_string:
        _configure_appinsights(appinsights_connection_string, service_name)


def _configure_appinsights(connection_string: str, service_name: str) -> None:
    """Attach Azure Monitor OpenTelemetry exporter."""
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=connection_string)
    except ImportError:
        logging.getLogger(__name__).warning(
            "azure-monitor-opentelemetry not installed — App Insights export disabled"
        )


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(**context)


configure_logging()
