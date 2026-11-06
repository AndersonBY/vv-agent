"""Provider payloads shared by classification tests and the kernel producer."""

CONTEXT_QUOTA_CASES = [
    ("quota_day", 429, {"message": "too many tokens per day"}, False),
    ("quota_minute", 400, {"message": "too many tokens per minute"}, False),
    ("quota_tpm", 429, {"message": "Request too large on tokens per min (TPM)"}, False),
    ("per_request", 400, {"message": "too many tokens per request"}, True),
    ("context_quota", 400, {"code": "context_length_exceeded", "message": "quota"}, True),
    ("context_429", 429, {"code": "context_length_exceeded"}, True),
    ("model_window", 400, {"message": "Model context window is 128k tokens, you requested 256k tokens."}, True),
    ("window_exceeded", 400, {"message": "context window exceeded"}, True),
    ("model_maximum", 400, {"message": "too large for model with 128k maximum context length"}, True),
]
