"""
Slack webhook notification helper.

Usage::

    from aerocast.notify import send_slack
    send_slack(":white_check_mark: Ingestion complete — 1 240 rows saved.")

The webhook URL is read from SLACK_WEBHOOK_URL in settings (or .env).
Pass ``webhook_url`` explicitly to override.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


def send_slack(
    message: str,
    webhook_url: str = "",
    timeout: int = 10,
) -> bool:
    """
    Post *message* to the configured Slack webhook.

    Args:
        message:     Slack-formatted text (supports mrkdwn).
        webhook_url: Override for SLACK_WEBHOOK_URL in settings.
        timeout:     HTTP request timeout in seconds.

    Returns:
        True on success, False on any error (never raises).
    """
    from aerocast.config import settings

    url = webhook_url or settings.slack_webhook_url
    if not url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return False

    try:
        resp = requests.post(url, json={"text": message}, timeout=timeout)
        resp.raise_for_status()
        logger.info("Slack notification sent.")
        return True
    except Exception as exc:
        logger.error("Slack notification failed: %s", exc)
        return False
