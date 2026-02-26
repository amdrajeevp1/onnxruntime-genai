"""ONNX Runtime GenAI Python package."""

from importlib.metadata import PackageNotFoundError, version
from onnxruntime_genai import _dll_directory

try:
    __version__ = version("onnxruntime-genai")
except PackageNotFoundError:
    __version__ = "0.0.0"

_dll_directory.add_onnxruntime_dependency("onnxruntime-genai")
