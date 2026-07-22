"""Example settings file for vv-agent.

Copy this file to `local_settings.py` and fill in your real API keys.
"""

LLM_SETTINGS = {
    "VERSION": "2",
    "backends": {
        "moonshot": {
            "default_endpoint": "moonshot-official",
            "models": {
                "kimi-k3": {
                    "id": "kimi-k3",
                    "endpoints": [
                        {
                            "endpoint_id": "moonshot-official",
                            "model_id": "kimi-k3",
                        }
                    ],
                    "context_length": 1048576,
                    "max_output_tokens": 131072,
                    "native_multimodal": True,
                    "function_call_available": True,
                    "response_format_available": True,
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
