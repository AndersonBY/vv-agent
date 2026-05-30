"""Example settings file for vv-agent.

Copy this file to `local_settings.py` and fill in your real API keys.
"""

LLM_SETTINGS = {
    "VERSION": "2",
    "backends": {
        "moonshot": {
            "default_endpoint": "moonshot-official",
            "models": {
                "kimi-k2.6": {
                    "id": "kimi-k2.6",
                    "endpoints": [
                        {
                            "endpoint_id": "moonshot-official",
                            "model_id": "kimi-k2.6",
                        }
                    ],
                },
            },
        },
        "openai": {
            "default_endpoint": "openai-official",
            "models": {
                "gpt-5.5": {
                    "id": "gpt-5.5",
                    "endpoints": ["openai-official"],
                }
            },
        },
    },
    "endpoints": [
        {
            "id": "moonshot-official",
            "api_key": "REPLACE_WITH_MOONSHOT_API_KEY",
            "api_base": "https://api.moonshot.cn/v1",
            "endpoint_type": "openai",
        },
        {
            "id": "openai-official",
            "api_key": "REPLACE_WITH_OPENAI_API_KEY",
            "api_base": "https://api.openai.com/v1",
            "endpoint_type": "openai",
        },
    ],
}

# Optional: default model/backend for memory-compression summary.
# Priority in runtime:
# 1) AgentTask.metadata (memory_summary_backend / memory_summary_model)
# 2) local_settings.py constants below
# 3) fallback to runtime default_backend + task.model
DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND = "moonshot"
DEFAULT_USER_MEMORY_SUMMARIZE_MODEL = "kimi-k2.6"
