import os

import pytest
from hypothesis import settings

from ast_optimizer.input_layer.ast_builder import ASTBuilder

settings.register_profile("synthesis", max_examples=100, deadline=None)
settings.register_profile(
    "ci",
    parent=settings.get_profile("synthesis"),
    derandomize=True,
    database=None,
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "synthesis"))


@pytest.fixture
def builder():
    return ASTBuilder()