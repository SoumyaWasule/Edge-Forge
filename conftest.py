"""Root conftest — ensures pytest can resolve bare imports from repo root."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
