import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from yoxla.inference.adapters import (
    OpenAICompatibleAdapter,
    TransformersAdapter,
)
from yoxla.inference.base import ModelCapabilities


def get_default_config_path():
    """
    Return the providers.yaml bundled with the YOXLA package.
    """
    return files("yoxla.configs").joinpath("providers.yaml")


def load_providers_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load provider configuration.

    If config_path is None, use the providers.yaml bundled
    inside the installed YOXLA package.
    """

    if config_path is None:
        config_resource = get_default_config_path()

        with config_resource.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

    else:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Providers config not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            "Providers config is empty."
        )

    if "providers" not in config:
        raise ValueError(
            "Config must contain a 'providers' section."
        )

    providers = config["providers"]

    if not isinstance(providers, dict):
        raise ValueError(
            "'providers' must be a mapping."
        )

    return providers


def create_model(
    provider_name: str,
    model: str,
    config_path: str | Path | None = None,
    model_options: dict[str, Any] | None = None,
):
    """
    Create a model adapter from provider configuration.

    Parameters
    ----------
    provider_name:
        Provider alias defined in providers.yaml.

    model:
        Provider model ID or local model path.

    config_path:
        Optional custom providers.yaml.
        If omitted, use the config bundled with YOXLA.

    model_options:
        Runtime model options, primarily for local models.
    """

    providers = load_providers_config(
        config_path=config_path
    )

    if provider_name not in providers:
        available = ", ".join(
            sorted(providers.keys())
        )

        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {available}"
        )

    config = providers[provider_name]

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid configuration for provider "
            f"'{provider_name}'."
        )

    backend = config.get("backend")

    if not backend:
        raise ValueError(
            f"Provider '{provider_name}' does not "
            f"define a backend."
        )

    model_options = model_options or {}

    capabilities = ModelCapabilities(
        **config.get("capabilities", {})
    )

    # --------------------------------------------------------------
    # OpenAI-compatible providers
    # --------------------------------------------------------------

    if backend == "openai_compatible":
        api_key = config.get("api_key")

        if api_key is None:
            api_key_env = config.get(
                "api_key_env"
            )

            if not api_key_env:
                raise ValueError(
                    f"Provider '{provider_name}' requires "
                    f"'api_key' or 'api_key_env'."
                )

            api_key = os.environ.get(
                api_key_env
            )

            if not api_key:
                raise ValueError(
                    f"Environment variable "
                    f"'{api_key_env}' is not set."
                )

        return OpenAICompatibleAdapter(
            model=model,
            provider=provider_name,
            api_key=api_key,
            base_url=config.get(
                "base_url"
            ),
            capabilities=capabilities,
            default_headers=config.get(
                "default_headers"
            ),
            max_tokens_param=config.get(
                "max_tokens_param",
                "max_tokens",
            ),
            thinking_control=config.get(
                "thinking_control"
            ),
        )

    # --------------------------------------------------------------
    # Hugging Face Transformers
    # --------------------------------------------------------------

    if backend == "transformers":
        return TransformersAdapter(
            model=model,

            device_map=model_options.get(
                "device_map",
                config.get(
                    "device_map",
                    "auto",
                ),
            ),

            dtype=model_options.get(
                "dtype",
                config.get(
                    "dtype",
                    "auto",
                ),
            ),

            trust_remote_code=model_options.get(
                "trust_remote_code",
                config.get(
                    "trust_remote_code",
                    False,
                ),
            ),

            load_in_4bit=model_options.get(
                "load_in_4bit",
                False,
            ),

            load_in_8bit=model_options.get(
                "load_in_8bit",
                False,
            ),

            capabilities=capabilities,
        )

    raise ValueError(
        f"Unsupported backend "
        f"'{backend}' for provider "
        f"'{provider_name}'."
    )