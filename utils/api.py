from dataclasses import dataclass

@dataclass
class APIModelConfig:
    start_system: str
    end_system: str
    start_user: str
    end_user: str
    start_assistant: str

QwenConfig = APIModelConfig(
    start_system='<|im_start|>system\n',
    end_system='<|im_end|>\n',
    start_user='<|im_start|>user\n',
    end_user='<|im_end|>\n',
    start_assistant='<|im_start|>assistant\n',
)

API_CONFIGS = {
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": QwenConfig,
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct": QwenConfig,
}
