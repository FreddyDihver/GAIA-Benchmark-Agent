from google.api_core.exceptions import (ResourceExhausted,
                                        ServiceUnavailable,
                                        TooManyRequests,
                                        InternalServerError)
from langchain_google_genai.chat_models import GoogleRateLimitError
import time

RETRYABLE_ERRORS = (
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
    InternalServerError,
    GoogleRateLimitError
)

class RetryLLM:
    def __init__(self, model, max_retries=3, wait_seconds=60):
        self.model = model
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds

    def bind_tools(self, tools, **kwargs):
        return RetryRunnable(
            self.model.bind_tools(tools, **kwargs),
            self.max_retries,
            self.wait_seconds,
        )


class RetryRunnable:
    def __init__(self, runnable, max_retries, wait_seconds):
        self.runnable = runnable
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds

    def invoke(self, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return self.runnable.invoke(*args, **kwargs)
            except RETRYABLE_ERRORS:
                if attempt == self.max_retries:
                    raise

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {self.wait_seconds:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(self.wait_seconds)