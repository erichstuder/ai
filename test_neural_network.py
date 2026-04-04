from neural_network import NeuralNetwork

def test_dummy():
    net = NeuralNetwork()
    net.train()
    result = net.query()
    assert result == "Query does nothing yet, but hello anyway!"
