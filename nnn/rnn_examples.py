"""
Examples demonstrating recurrent neural network layers.

This file shows how to use:
- RNNCell and RNN for basic recurrent networks
- LSTMCell and LSTM for Long Short-Term Memory
- GRUCell and GRU for Gated Recurrent Units

Sequence format: (seq_len, batch, features) unless batch_first=True
"""

import numpy as np
from nnn import Tensor, Dense
from nnn.rnn import RNNCell, RNN, LSTMCell, LSTM, GRUCell, GRU
from nnn.activations import softmax
from nnn.optim import Adam
from nnn.losses import mse_loss, cross_entropy_loss


def example_rnn_cell():
    """Example: Basic RNN Cell."""
    print("=" * 60)
    print("Basic RNN Cell")
    print("=" * 60)
    
    # Create RNN cell: input_size=10, hidden_size=20
    cell = RNNCell(input_size=10, hidden_size=20)
    
    print(f"Layer: {cell}")
    print(f"W_ih shape: {cell.W_ih.shape} (input to hidden)")
    print(f"W_hh shape: {cell.W_hh.shape} (hidden to hidden)")
    print(f"Bias shape: {cell.bias.shape}")
    
    # Single time step
    batch_size = 4
    x_t = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    
    print(f"\nInput shape: {x_t.shape}")
    
    # First step (no previous hidden state)
    h_1 = cell(x_t)
    print(f"Output h_1 shape: {h_1.shape}")
    
    # Second step (with previous hidden state)
    x_t2 = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    h_2 = cell(x_t2, h_1)
    print(f"Output h_2 shape: {h_2.shape}")
    
    print("\nFormula: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b)")
    print()


def example_rnn_sequence():
    """Example: RNN processing a sequence."""
    print("=" * 60)
    print("RNN Processing Sequence")
    print("=" * 60)
    
    # Create RNN: 1 layer
    rnn = RNN(input_size=10, hidden_size=20, num_layers=1)
    
    print(f"Layer: {rnn}")
    
    # Input sequence: (seq_len, batch, input_size)
    seq_len = 5
    batch_size = 4
    x = Tensor(np.random.randn(seq_len, batch_size, 10).astype(np.float32))
    
    print(f"\nInput shape: {x.shape} (seq_len, batch, features)")
    
    # Forward pass
    outputs, h_n = rnn(x)
    
    print(f"Outputs shape: {outputs.shape} (all hidden states)")
    print(f"h_n shape: {h_n.shape} (final hidden state)")
    print()


def example_rnn_multilayer():
    """Example: Multi-layer RNN."""
    print("=" * 60)
    print("Multi-layer RNN")
    print("=" * 60)
    
    # Create 3-layer RNN
    rnn = RNN(input_size=10, hidden_size=20, num_layers=3)
    
    print(f"Layer: {rnn}")
    print(f"Number of parameters: {sum(p.data.size for p in rnn.parameters())}")
    
    # Input sequence
    x = Tensor(np.random.randn(5, 4, 10).astype(np.float32))
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    outputs, h_n = rnn(x)
    
    print(f"Outputs shape: {outputs.shape}")
    print(f"  (outputs from last layer at each time step)")
    print(f"h_n shape: {h_n.shape}")
    print(f"  (final hidden state from each layer)")
    print()


def example_rnn_batch_first():
    """Example: RNN with batch_first=True."""
    print("=" * 60)
    print("RNN with batch_first=True")
    print("=" * 60)
    
    # Create RNN with batch_first
    rnn = RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
    
    print(f"Layer: {rnn}")
    
    # Input: (batch, seq_len, features) instead of (seq_len, batch, features)
    batch_size = 4
    seq_len = 5
    x = Tensor(np.random.randn(batch_size, seq_len, 10).astype(np.float32))
    
    print(f"\nInput shape: {x.shape} (batch, seq_len, features)")
    
    outputs, h_n = rnn(x)
    
    print(f"Outputs shape: {outputs.shape} (batch, seq_len, hidden)")
    print(f"h_n shape: {h_n.shape}")
    print()


def example_lstm_cell():
    """Example: LSTM Cell with gates."""
    print("=" * 60)
    print("LSTM Cell")
    print("=" * 60)
    
    # Create LSTM cell
    cell = LSTMCell(input_size=10, hidden_size=20)
    
    print(f"Layer: {cell}")
    print(f"W_ih shape: {cell.W_ih.shape} (4 gates combined)")
    print(f"W_hh shape: {cell.W_hh.shape} (4 gates combined)")
    print(f"  Each weight has 4x hidden_size for: input, forget, cell, output gates")
    
    # Single time step
    batch_size = 4
    x_t = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    
    print(f"\nInput shape: {x_t.shape}")
    
    # First step (no previous state)
    h_1, c_1 = cell(x_t)
    print(f"Output h_1 shape: {h_1.shape} (hidden state)")
    print(f"Output c_1 shape: {c_1.shape} (cell state)")
    
    # Second step (with previous state)
    x_t2 = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    h_2, c_2 = cell(x_t2, (h_1, c_1))
    print(f"Output h_2 shape: {h_2.shape}")
    print(f"Output c_2 shape: {c_2.shape}")
    
    print("\nLSTM Gates:")
    print("  i_t = sigmoid(W_ii @ x + W_hi @ h + b_i)  # Input gate")
    print("  f_t = sigmoid(W_if @ x + W_hf @ h + b_f)  # Forget gate")
    print("  g_t = tanh(W_ig @ x + W_hg @ h + b_g)     # Cell gate")
    print("  o_t = sigmoid(W_io @ x + W_ho @ h + b_o)  # Output gate")
    print("  c_t = f_t * c_{t-1} + i_t * g_t           # Cell state")
    print("  h_t = o_t * tanh(c_t)                     # Hidden state")
    print()


def example_lstm_sequence():
    """Example: LSTM processing a sequence."""
    print("=" * 60)
    print("LSTM Processing Sequence")
    print("=" * 60)
    
    # Create 2-layer LSTM
    lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)
    
    print(f"Layer: {lstm}")
    
    # Input sequence
    seq_len = 8
    batch_size = 4
    x = Tensor(np.random.randn(seq_len, batch_size, 10).astype(np.float32))
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    outputs, (h_n, c_n) = lstm(x)
    
    print(f"Outputs shape: {outputs.shape}")
    print(f"h_n shape: {h_n.shape} (final hidden for each layer)")
    print(f"c_n shape: {c_n.shape} (final cell for each layer)")
    print()


def example_gru_cell():
    """Example: GRU Cell."""
    print("=" * 60)
    print("GRU Cell")
    print("=" * 60)
    
    # Create GRU cell
    cell = GRUCell(input_size=10, hidden_size=20)
    
    print(f"Layer: {cell}")
    print(f"W_ih shape: {cell.W_ih.shape} (3 gates combined)")
    print(f"W_hh shape: {cell.W_hh.shape} (3 gates combined)")
    print(f"  Each weight has 3x hidden_size for: reset, update, new gates")
    
    # Single time step
    batch_size = 4
    x_t = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    
    print(f"\nInput shape: {x_t.shape}")
    
    # Forward pass
    h_1 = cell(x_t)
    print(f"Output h_1 shape: {h_1.shape}")
    
    # Second step
    x_t2 = Tensor(np.random.randn(batch_size, 10).astype(np.float32))
    h_2 = cell(x_t2, h_1)
    print(f"Output h_2 shape: {h_2.shape}")
    
    print("\nGRU Gates (simpler than LSTM - no separate cell state):")
    print("  r_t = sigmoid(W_ir @ x + W_hr @ h + b_r)       # Reset gate")
    print("  z_t = sigmoid(W_iz @ x + W_hz @ h + b_z)       # Update gate")
    print("  n_t = tanh(W_in @ x + r_t * (W_hn @ h) + b_n)  # New gate")
    print("  h_t = (1 - z_t) * n_t + z_t * h_{t-1}          # Hidden state")
    print()


def example_gru_sequence():
    """Example: GRU processing a sequence."""
    print("=" * 60)
    print("GRU Processing Sequence")
    print("=" * 60)
    
    # Create 2-layer GRU
    gru = GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    
    print(f"Layer: {gru}")
    
    # Input sequence (batch_first)
    batch_size = 4
    seq_len = 8
    x = Tensor(np.random.randn(batch_size, seq_len, 10).astype(np.float32))
    
    print(f"\nInput shape: {x.shape} (batch, seq_len, features)")
    
    # Forward pass
    outputs, h_n = gru(x)
    
    print(f"Outputs shape: {outputs.shape}")
    print(f"h_n shape: {h_n.shape}")
    print()


def example_rnn_comparison():
    """Example: Compare RNN, LSTM, GRU."""
    print("=" * 60)
    print("RNN vs LSTM vs GRU Comparison")
    print("=" * 60)
    
    input_size = 10
    hidden_size = 20
    num_layers = 2
    
    rnn = RNN(input_size, hidden_size, num_layers)
    lstm = LSTM(input_size, hidden_size, num_layers)
    gru = GRU(input_size, hidden_size, num_layers)
    
    # Count parameters
    rnn_params = sum(p.data.size for p in rnn.parameters())
    lstm_params = sum(p.data.size for p in lstm.parameters())
    gru_params = sum(p.data.size for p in gru.parameters())
    
    print(f"Configuration: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    print()
    print(f"{'Model':<10} {'Parameters':<15} {'Gates':<20} {'Cell State':<10}")
    print("-" * 55)
    print(f"{'RNN':<10} {rnn_params:<15} {'None':<20} {'No':<10}")
    print(f"{'GRU':<10} {gru_params:<15} {'Reset, Update':<20} {'No':<10}")
    print(f"{'LSTM':<10} {lstm_params:<15} {'Input, Forget,':<20} {'Yes':<10}")
    print(f"{'':10} {'':15} {'Cell, Output':<20} {'':10}")
    print()
    print("Notes:")
    print("  - RNN: Simple but suffers from vanishing gradients")
    print("  - LSTM: Best for long sequences, most parameters")
    print("  - GRU: Good balance, fewer parameters than LSTM")
    print()


def example_sequence_classification():
    """Example: Sequence classification with LSTM."""
    print("=" * 60)
    print("Sequence Classification with LSTM")
    print("=" * 60)
    
    # Task: Classify a sequence of 10 time steps into 3 classes
    seq_len = 10
    batch_size = 8
    input_size = 5
    hidden_size = 16
    num_classes = 3
    
    # Create model: LSTM + Dense classifier
    lstm = LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    classifier = Dense(hidden_size, num_classes)
    
    print("Architecture:")
    print(f"  Input: (batch, {seq_len}, {input_size})")
    print(f"  LSTM: hidden_size={hidden_size}")
    print(f"  Dense: {hidden_size} -> {num_classes}")
    print()
    
    # Create dummy data
    np.random.seed(42)
    x = Tensor(np.random.randn(batch_size, seq_len, input_size).astype(np.float32), 
               requires_grad=True)
    targets = np.random.randint(0, num_classes, batch_size)
    y = Tensor(targets.astype(np.float32))
    
    print(f"Input shape: {x.shape}")
    print(f"Targets: {targets}")
    
    # Forward pass
    outputs, (h_n, c_n) = lstm(x)
    
    # Use final hidden state for classification
    # h_n shape: (num_layers, batch, hidden) -> take last layer
    final_hidden = Tensor(h_n.data[-1])
    print(f"\nFinal hidden shape: {final_hidden.shape}")
    
    # Classify
    logits = classifier(final_hidden)
    print(f"Logits shape: {logits.shape}")
    
    # Get predictions
    predictions = np.argmax(logits.data, axis=1)
    print(f"Predictions: {predictions}")
    print()


def example_sequence_to_sequence():
    """Example: Sequence-to-sequence with RNN."""
    print("=" * 60)
    print("Sequence-to-Sequence (Many-to-Many)")
    print("=" * 60)
    
    # Task: Process each time step and produce output at each step
    seq_len = 6
    batch_size = 4
    input_size = 8
    hidden_size = 16
    output_size = 4
    
    # Model: GRU + Dense at each step
    gru = GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    output_layer = Dense(hidden_size, output_size)
    
    print("Architecture:")
    print(f"  Input: (batch, {seq_len}, {input_size})")
    print(f"  GRU outputs at each step: (batch, {seq_len}, {hidden_size})")
    print(f"  Dense applied to each step: (batch, {seq_len}, {output_size})")
    print()
    
    # Create data
    x = Tensor(np.random.randn(batch_size, seq_len, input_size).astype(np.float32))
    
    # Forward pass
    outputs, h_n = gru(x)
    print(f"GRU outputs shape: {outputs.shape}")
    
    # Apply dense layer to each time step
    # Reshape: (batch, seq, hidden) -> (batch * seq, hidden)
    outputs_flat = Tensor(outputs.data.reshape(-1, hidden_size))
    logits_flat = output_layer(outputs_flat)
    
    # Reshape back: (batch * seq, output) -> (batch, seq, output)
    logits = Tensor(logits_flat.data.reshape(batch_size, seq_len, output_size))
    
    print(f"Final outputs shape: {logits.shape}")
    print()


def example_rnn_training():
    """Example: Training an RNN."""
    print("=" * 60)
    print("RNN Training Example")
    print("=" * 60)
    
    # Simple task: Learn to output the mean of input sequence
    np.random.seed(42)
    
    # Model
    rnn = RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
    fc = Dense(10, 1)
    
    # Optimizer
    params = rnn.parameters() + fc.parameters()
    optimizer = Adam(params, lr=0.01)
    
    print("Task: Predict mean of input sequence")
    print("Model: RNN(1, 10) -> Dense(10, 1)")
    print()
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        
        for _ in range(10):
            # Generate batch: sequences of length 5
            batch_size = 8
            seq_len = 5
            x_data = np.random.randn(batch_size, seq_len, 1).astype(np.float32)
            y_data = np.mean(x_data, axis=1)  # Target: mean of sequence
            
            x = Tensor(x_data, requires_grad=True)
            y = Tensor(y_data)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, h_n = rnn(x)
            final_hidden = Tensor(h_n.data[-1])
            pred = fc(final_hidden)
            
            # Loss
            loss = mse_loss(pred, y)
            total_loss += loss.data.item()
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
        
        avg_loss = total_loss / 10
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    print()


if __name__ == "__main__":
    # Run all examples
    example_rnn_cell()
    example_rnn_sequence()
    example_rnn_multilayer()
    example_rnn_batch_first()
    example_lstm_cell()
    example_lstm_sequence()
    example_gru_cell()
    example_gru_sequence()
    example_rnn_comparison()
    example_sequence_classification()
    example_sequence_to_sequence()
    example_rnn_training()
    
    print("=" * 60)
    print("All RNN examples completed!")
    print("=" * 60)
