import logging
import sys

import structlog

from api.config import settings


def _add_app_name(logger, method_name, event_dict):
    event_dict["app"] = settings.APP_NAME
    return event_dict


def configure_logging() -> None:
    log_level_name = "DEBUG" if settings.DEBUG else settings.LOG_LEVEL.upper()
    log_level = logging.getLevelNamesMapping()[log_level_name]

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_name,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.APP_ENV == "dev":
        final_processor = structlog.dev.ConsoleRenderer(colors=True, pad_level=False)
        exc_processor = structlog.processors.format_exc_info
    else:
        final_processor = structlog.processors.JSONRenderer()
        exc_processor = structlog.processors.dict_tracebacks

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            exc_processor,
            final_processor,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.propagate = False
        uv_logger.setLevel(log_level)

    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
