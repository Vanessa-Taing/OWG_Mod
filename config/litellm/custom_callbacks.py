# my_custom_logger.py
import json
import os
import time
from litellm.integrations.custom_logger import CustomLogger
import litellm
from datetime import datetime

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "litellm_logs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)


def write_log(entry):
    def default(o):
        try:
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)
        except Exception:
            return str(o)

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=default) + "\n")


class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        # optional: you can print or track requests here
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "anonymous")

        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})

        # Compute cost safely
        try:
            cost = litellm.completion_cost(completion_response=response_obj)
        except Exception:
            cost = 0.0

        # Extract token usage safely
        usage = response_obj.get("usage", {}) if isinstance(response_obj, dict) else {}

        # Extract text output safely
        try:
            content = response_obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = ""

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
            "model": model,
            "user": user,
            "cost": cost,
            "response": content,
            "usage": usage,
            "messages": messages,
            "metadata": metadata,
        }

        write_log(log_entry)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "unknown")
        user = kwargs.get("user", "anonymous")
        exception_event = kwargs.get("exception", None)

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failure",
            "model": model,
            "user": user,
            "cost": 0.0,
            "response": str(exception_event),
        }

        write_log(log_entry)


# Register your handler
proxy_handler_instance = MyCustomHandler()
