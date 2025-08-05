"""Util classes to load models from different providers."""

from copy import deepcopy
from typing import Any

from .model_scaffold import Huggingface, Model, RecurrentGemmaKaggle


IMPLEMENTATION_KWARGS = {
    "huggingface": ["max_tokens", "dtype"],
    "recurrentgemma": ["max_tokens", "dtype", "device"],
}
IMPLEMENTATION_CLASSES = {
    "huggingface": Huggingface,
    "recurrentgemma": RecurrentGemmaKaggle,
}


def kwargs_cleaner(model_type: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
    """Clean kwargs dependent on specified model type.

    Args:
        model_type (str): Type of the model to clean for.
        kwargs (dict[str, Any]): Kwargs to be cleaned.

    Returns:
        dict[str, Any]: Cleaned kwargs.

    """
    kwargs_return = deepcopy(kwargs)
    for key, value in kwargs.items():
        del_keys = [
            key for key, value in kwargs.items() if value is None or key not in IMPLEMENTATION_KWARGS[model_type]
        ]
        for key in del_keys:
            del kwargs_return[key]
    return kwargs_return


def load_model(model_type: str, model_path: str, tokenizer_path: str | None = None, **kwargs) -> Model:
    """Load a scaffolded version of implemented models with generic kwargs.

    Args:
        model_type (str): Class type of the model.
        model_path (str): Path to model, used to identify correct tensors.
        tokenizer_path (str | None, optional): Path to tokenizer, used to identify
            correct tokenizer object. Defaults to None.
        **kwargs (dict): Generic kwargs to use during loading.

    Returns:
        Model: Loaded instance of the scaffolded model.

    """
    if tokenizer_path is None:
        tokenizer_path = model_path
    clean_kwargs = kwargs_cleaner(model_type, **kwargs)
    return IMPLEMENTATION_CLASSES[model_type](model_path, tokenizer_path, **clean_kwargs)
