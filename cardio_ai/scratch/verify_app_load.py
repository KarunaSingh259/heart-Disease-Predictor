import torch
import sys
import os
sys.path.append(os.getcwd())

# Mocking flask for standalone test
class MockFlask:
    def __init__(self, *args, **kwargs): pass
    def route(self, *args, **kwargs): return lambda x: x
    def secret_key(self, *args, **kwargs): pass

import flask
flask.Flask = MockFlask
flask.session = {}

# Import the model loading logic
from app import echo_model

if echo_model is not None:
    print("Success: echo_model loaded successfully from app.py")
    # Test with dummy input (1, 3, 16, 112, 112)
    dummy_input = torch.zeros((1, 3, 16, 112, 112))
    with torch.no_grad():
        output = echo_model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Model keys: {list(echo_model.state_dict().keys())[:5]}")
else:
    print("Failure: echo_model is None")
