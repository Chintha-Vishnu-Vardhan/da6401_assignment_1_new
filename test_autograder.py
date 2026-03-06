import numpy as np
from src.ann.neural_network import NeuralNetwork

def test_autograder_compliance():
    print("Running Autograder Compliance Checks...\n")
    
    # 1. Test CLI setup (dummy args)
    class DummyArgs:
        activation = "relu"
        loss = "mse"
        optimizer = "sgd"
        learning_rate = 0.01
        weight_decay = 0.0
        hidden_size = [64, 64]
        num_layers = 2
        
    args = DummyArgs()
    
    try:
        model = NeuralNetwork(args, input_dim=784, num_classes=10)
        print("✅ Model Initialization Passed")
    except Exception as e:
        print(f"❌ Model Initialization Failed: {e}")
        return

    # 2. Test get_weights and set_weights methods
    try:
        weights = model.get_weights()
        model.set_weights(weights)
        print("✅ get_weights() and set_weights() Passed")
    except AttributeError:
        print("❌ FAILED: get_weights() or set_weights() is missing from NeuralNetwork!")
    except Exception as e:
        print(f"❌ FAILED: Weights shape mismatch or error: {e}")

    # 3. Test Backward Gradient Order (Must be Last to First)
    try:
        # Dummy data for a forward/backward pass
        X_dummy = np.random.randn(5, 784)
        y_dummy = np.zeros((5, 10))
        y_dummy[0, 1] = 1 
        
        logits = model.forward(X_dummy)
        loss, probs = model.compute_loss_and_output(y_dummy)
        grad_W_list, grad_b_list = model.backward(y_dummy, probs)
        
        # The output layer W shape should be (64, 10).
        # If the gradients are correctly ordered (Last to First), grad_W_list[0] will be (64, 10).
        if grad_W_list[0].shape == (64, 10):
            print("✅ Backward Gradient Order Passed (Last layer to First layer)")
        else:
            print(f"❌ FAILED: Gradient order is wrong. Expected first matrix shape (64, 10), got {grad_W_list[0].shape}. You are returning First to Last!")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")

if __name__ == "__main__":
    test_autograder_compliance()