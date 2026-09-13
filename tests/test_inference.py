from yoxla.inference.base import ModelAdapter
from yoxla.inference.schemas import (
    GenerationRequest,
    GenerationResponse,
    Message,
)


class DummyAdapter(ModelAdapter):

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        return GenerationResponse(
            text="Bakı",
            model=self.model,
            provider=self.provider,
        )


def test_model_adapter():
    model = DummyAdapter(
        model="dummy-model",
        provider="test",
    )

    request = GenerationRequest(
        messages=[
            Message(
                role="user",
                content="Azərbaycanın paytaxtı hansıdır?",
            )
        ]
    )

    response = model.generate(request)

    assert response.text == "Bakı"
    assert response.model == "dummy-model"
    assert response.provider == "test"