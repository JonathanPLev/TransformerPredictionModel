
import torch
from neural_network.mini_nn import MiniNN


def test_mini_nn_init():
    model = MiniNN(input_dim=50, hidden_dim=32)
    
    assert isinstance(model, torch.nn.Module)
    assert model.model[0].in_features == 50
    assert model.model[-1].out_features == 2


def test_mini_nn_forward():
    model = MiniNN(input_dim=50, hidden_dim=32)
    
    x = torch.randn(10, 50)
    output = model(x)
    
    assert output.shape == (10, 2)
    assert isinstance(output, torch.Tensor)


def test_mini_nn_output_probabilities():
    model = MiniNN(input_dim=50, hidden_dim=32)
    
    x = torch.randn(5, 50)
    output = model(x)
    
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-6)


def test_mini_nn_batch_sizes():
    model = MiniNN(input_dim=50)
    
    for batch_size in [1, 16, 32]:
        x = torch.randn(batch_size, 50)
        output = model(x)
        assert output.shape == (batch_size, 2)