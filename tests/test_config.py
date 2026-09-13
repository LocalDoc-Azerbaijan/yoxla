from yoxla.inference import load_providers_config


def test_bundled_providers_config():
    providers = load_providers_config()

    assert "openrouter" in providers
    assert "google" in providers
    assert "openai" in providers
    assert "vllm" in providers
    assert "llamacpp" in providers
    assert "transformers" in providers