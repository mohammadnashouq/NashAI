"""
Recurrent neural network layers.

This module implements RNN components including:
- RNNCell / RNN: Basic recurrent units
- LSTMCell / LSTM: Long Short-Term Memory
- GRUCell / GRU: Gated Recurrent Units

Key design considerations:
- Gradient flow: LSTM/GRU gates prevent vanishing gradients
- Sequence format: (seq_len, batch, features)
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .tensor import Tensor
from .activations import tanh, sigmoid


class RNNCell:
    """
    Basic RNN Cell.
    
    Computes: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    Input shape: (batch, input_size)
    Hidden shape: (batch, hidden_size)
    Output shape: (batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True
    ):
        """
        Initialize RNN cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            use_bias: Whether to use bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # Xavier initialization
        limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))
        
        # Input-to-hidden weights
        self.W_ih = Tensor(
            np.random.uniform(-limit_ih, limit_ih, (input_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Hidden-to-hidden weights
        self.W_hh = Tensor(
            np.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Bias
        if use_bias:
            self.bias = Tensor(
                np.zeros(hidden_size, dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias = None
    
    def __call__(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            h: Previous hidden state of shape (batch, hidden_size)
            
        Returns:
            New hidden state of shape (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        # Compute: h_new = tanh(x @ W_ih + h @ W_hh + bias)
        out = x @ self.W_ih + h @ self.W_hh
        
        if self.use_bias:
            out = out + self.bias
        
        h_new = tanh(out)
        
        return h_new
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = [self.W_ih, self.W_hh]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset gradients."""
        self.W_ih.grad = None
        self.W_hh.grad = None
        if self.bias is not None:
            self.bias.grad = None
    
    def __repr__(self) -> str:
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class RNN:
    """
    Multi-layer RNN.
    
    Processes sequences using stacked RNN cells.
    
    Input shape: (seq_len, batch, input_size)
    Output shape: (seq_len, batch, hidden_size), (batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_bias: bool = True,
        batch_first: bool = False
    ):
        """
        Initialize RNN.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked RNN layers
            use_bias: Whether to use bias
            batch_first: If True, input shape is (batch, seq, features)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create RNN cells for each layer
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(RNNCell(cell_input_size, hidden_size, use_bias))
    
    def __call__(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all time steps.
        
        Args:
            x: Input tensor of shape (seq_len, batch, input_size) or 
               (batch, seq_len, input_size) if batch_first=True
            h_0: Initial hidden state of shape (num_layers, batch, hidden_size)
            
        Returns:
            outputs: All hidden states (seq_len, batch, hidden_size)
            h_n: Final hidden state (num_layers, batch, hidden_size)
        """
        # Handle batch_first
        if self.batch_first:
            # (batch, seq, features) -> (seq, batch, features)
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states
        if h_0 is None:
            h = [Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32)) 
                 for _ in range(self.num_layers)]
        else:
            h = [Tensor(h_0.data[i], requires_grad=h_0.requires_grad) 
                 for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            # Get input for this time step
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            # Pass through each layer
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                h[i] = cell(layer_input, h[i])
                layer_input = h[i]
            
            outputs.append(h[-1].data)
        
        # Stack outputs
        output_data = np.stack(outputs, axis=0)  # (seq_len, batch, hidden)
        
        # Stack final hidden states
        h_n_data = np.stack([h_i.data for h_i in h], axis=0)  # (num_layers, batch, hidden)
        
        # Handle batch_first for output
        if self.batch_first:
            output_data = output_data.transpose(1, 0, 2)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        h_n = Tensor(h_n_data, requires_grad=x.requires_grad)
        
        return output, h_n
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters from all cells."""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all cells."""
        for cell in self.cells:
            cell.zero_grad()
    
    def __repr__(self) -> str:
        return (f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers})")


class LSTMCell:
    """
    LSTM Cell with gating mechanism for better gradient flow.
    
    Gates:
        i_t = sigmoid(W_ii @ x + W_hi @ h + b_i)  # Input gate
        f_t = sigmoid(W_if @ x + W_hf @ h + b_f)  # Forget gate
        g_t = tanh(W_ig @ x + W_hg @ h + b_g)     # Cell gate
        o_t = sigmoid(W_io @ x + W_ho @ h + b_o)  # Output gate
    
    Cell state update:
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    
    Input shape: (batch, input_size)
    Hidden shape: (batch, hidden_size)
    Cell shape: (batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True
    ):
        """
        Initialize LSTM cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            use_bias: Whether to use bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # Combined weights for efficiency (4 gates)
        # Order: input, forget, cell, output
        limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))
        
        # Input-to-hidden weights for all gates: (input_size, 4 * hidden_size)
        self.W_ih = Tensor(
            np.random.uniform(-limit_ih, limit_ih, (input_size, 4 * hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Hidden-to-hidden weights for all gates: (hidden_size, 4 * hidden_size)
        self.W_hh = Tensor(
            np.random.uniform(-limit_hh, limit_hh, (hidden_size, 4 * hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Bias for all gates
        if use_bias:
            self.bias = Tensor(
                np.zeros(4 * hidden_size, dtype=np.float32),
                requires_grad=True
            )
            # Initialize forget gate bias to 1 for better gradient flow
            self.bias.data[hidden_size:2*hidden_size] = 1.0
        else:
            self.bias = None
        
        self._cache = {}
    
    def __call__(
        self,
        x: Tensor,
        hc: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            hc: Tuple of (hidden_state, cell_state), each (batch, hidden_size)
            
        Returns:
            Tuple of (new_hidden, new_cell), each (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize states if not provided
        if hc is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
            c = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        else:
            h, c = hc
        
        # Compute all gates at once
        gates = x @ self.W_ih + h @ self.W_hh
        
        if self.use_bias:
            gates = gates + self.bias
        
        # Split into individual gates
        gates_data = gates.data
        i_gate = gates_data[:, :self.hidden_size]
        f_gate = gates_data[:, self.hidden_size:2*self.hidden_size]
        g_gate = gates_data[:, 2*self.hidden_size:3*self.hidden_size]
        o_gate = gates_data[:, 3*self.hidden_size:]
        
        # Apply activations
        i = 1.0 / (1.0 + np.exp(-np.clip(i_gate, -500, 500)))  # sigmoid
        f = 1.0 / (1.0 + np.exp(-np.clip(f_gate, -500, 500)))  # sigmoid
        g = np.tanh(g_gate)
        o = 1.0 / (1.0 + np.exp(-np.clip(o_gate, -500, 500)))  # sigmoid
        
        # Update cell state: c_new = f * c + i * g
        c_new_data = f * c.data + i * g
        
        # Update hidden state: h_new = o * tanh(c_new)
        h_new_data = o * np.tanh(c_new_data)
        
        # Cache for backward
        self._cache['x'] = x
        self._cache['h'] = h
        self._cache['c'] = c
        self._cache['i'] = i
        self._cache['f'] = f
        self._cache['g'] = g
        self._cache['o'] = o
        self._cache['c_new'] = c_new_data
        self._cache['gates'] = gates
        
        # Create output tensors
        h_new = Tensor(
            h_new_data,
            requires_grad=x.requires_grad or (hc is not None and hc[0].requires_grad),
            _op='lstm_cell',
            _children=(x, h, c, self.W_ih, self.W_hh, self.bias) if self.use_bias else (x, h, c, self.W_ih, self.W_hh)
        )
        
        c_new = Tensor(
            c_new_data,
            requires_grad=x.requires_grad or (hc is not None and hc[1].requires_grad),
            _op='lstm_cell_c'
        )
        
        def _backward(grad_h):
            # This is a simplified backward - full BPTT would be more complex
            i, f, g, o = self._cache['i'], self._cache['f'], self._cache['g'], self._cache['o']
            c_new = self._cache['c_new']
            x, h, c = self._cache['x'], self._cache['h'], self._cache['c']
            
            tanh_c = np.tanh(c_new)
            
            # Gradient through output gate
            grad_o = grad_h * tanh_c
            grad_tanh_c = grad_h * o
            
            # Gradient through tanh
            grad_c_new = grad_tanh_c * (1 - tanh_c ** 2)
            
            # Gradients through cell state equation
            grad_f = grad_c_new * c.data
            grad_c = grad_c_new * f
            grad_i = grad_c_new * g
            grad_g = grad_c_new * i
            
            # Gradients through activations (sigmoid: s*(1-s), tanh: 1-t^2)
            grad_i_gate = grad_i * i * (1 - i)
            grad_f_gate = grad_f * f * (1 - f)
            grad_g_gate = grad_g * (1 - g ** 2)
            grad_o_gate = grad_o * o * (1 - o)
            
            # Concatenate gate gradients
            grad_gates = np.concatenate([grad_i_gate, grad_f_gate, grad_g_gate, grad_o_gate], axis=1)
            
            # Gradients w.r.t. weights
            if self.W_ih.requires_grad:
                grad_W_ih = x.data.T @ grad_gates
                if self.W_ih.grad is None:
                    self.W_ih.grad = Tensor(np.zeros_like(self.W_ih.data))
                self.W_ih.grad.data += grad_W_ih
            
            if self.W_hh.requires_grad:
                grad_W_hh = h.data.T @ grad_gates
                if self.W_hh.grad is None:
                    self.W_hh.grad = Tensor(np.zeros_like(self.W_hh.data))
                self.W_hh.grad.data += grad_W_hh
            
            if self.use_bias and self.bias.requires_grad:
                grad_bias = np.sum(grad_gates, axis=0)
                if self.bias.grad is None:
                    self.bias.grad = Tensor(np.zeros_like(self.bias.data))
                self.bias.grad.data += grad_bias
            
            # Gradient w.r.t. input
            grad_x = grad_gates @ self.W_ih.data.T if x.requires_grad else None
            grad_h_prev = grad_gates @ self.W_hh.data.T if h.requires_grad else None
            
            if self.use_bias:
                return (grad_x, grad_h_prev, grad_c, None, None, None)
            return (grad_x, grad_h_prev, grad_c, None, None)
        
        h_new._backward_fn = _backward
        
        return h_new, c_new
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = [self.W_ih, self.W_hh]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset gradients."""
        self.W_ih.grad = None
        self.W_hh.grad = None
        if self.bias is not None:
            self.bias.grad = None
    
    def __repr__(self) -> str:
        return f"LSTMCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class LSTM:
    """
    Multi-layer LSTM.
    
    Processes sequences using stacked LSTM cells.
    
    Input shape: (seq_len, batch, input_size)
    Output shape: (seq_len, batch, hidden_size), ((num_layers, batch, hidden), (num_layers, batch, hidden))
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_bias: bool = True,
        batch_first: bool = False
    ):
        """
        Initialize LSTM.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked LSTM layers
            use_bias: Whether to use bias
            batch_first: If True, input shape is (batch, seq, features)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create LSTM cells for each layer
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(cell_input_size, hidden_size, use_bias))
    
    def __call__(
        self,
        x: Tensor,
        hc_0: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through all time steps.
        
        Args:
            x: Input tensor of shape (seq_len, batch, input_size) or 
               (batch, seq_len, input_size) if batch_first=True
            hc_0: Tuple of (h_0, c_0), each of shape (num_layers, batch, hidden_size)
            
        Returns:
            outputs: All hidden states (seq_len, batch, hidden_size)
            (h_n, c_n): Final states, each (num_layers, batch, hidden_size)
        """
        # Handle batch_first
        if self.batch_first:
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize states
        if hc_0 is None:
            h = [Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32)) 
                 for _ in range(self.num_layers)]
            c = [Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32)) 
                 for _ in range(self.num_layers)]
        else:
            h_0, c_0 = hc_0
            h = [Tensor(h_0.data[i], requires_grad=h_0.requires_grad) 
                 for i in range(self.num_layers)]
            c = [Tensor(c_0.data[i], requires_grad=c_0.requires_grad) 
                 for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(layer_input, (h[i], c[i]))
                layer_input = h[i]
            
            outputs.append(h[-1].data)
        
        # Stack outputs
        output_data = np.stack(outputs, axis=0)
        h_n_data = np.stack([h_i.data for h_i in h], axis=0)
        c_n_data = np.stack([c_i.data for c_i in c], axis=0)
        
        if self.batch_first:
            output_data = output_data.transpose(1, 0, 2)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        h_n = Tensor(h_n_data, requires_grad=x.requires_grad)
        c_n = Tensor(c_n_data, requires_grad=x.requires_grad)
        
        return output, (h_n, c_n)
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters from all cells."""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all cells."""
        for cell in self.cells:
            cell.zero_grad()
    
    def __repr__(self) -> str:
        return (f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers})")


class GRUCell:
    """
    GRU Cell - simplified gating compared to LSTM.
    
    Gates:
        r_t = sigmoid(W_ir @ x + W_hr @ h + b_r)  # Reset gate
        z_t = sigmoid(W_iz @ x + W_hz @ h + b_z)  # Update gate
        n_t = tanh(W_in @ x + r_t * (W_hn @ h) + b_n)  # New gate
    
    Hidden state update:
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    
    Input shape: (batch, input_size)
    Hidden shape: (batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True
    ):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            use_bias: Whether to use bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # Combined weights for efficiency (3 gates: reset, update, new)
        limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))
        
        # Input-to-hidden weights for all gates: (input_size, 3 * hidden_size)
        self.W_ih = Tensor(
            np.random.uniform(-limit_ih, limit_ih, (input_size, 3 * hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Hidden-to-hidden weights for all gates: (hidden_size, 3 * hidden_size)
        self.W_hh = Tensor(
            np.random.uniform(-limit_hh, limit_hh, (hidden_size, 3 * hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Bias for all gates
        if use_bias:
            self.bias_ih = Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32),
                requires_grad=True
            )
            self.bias_hh = Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self._cache = {}
    
    def __call__(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            h: Previous hidden state of shape (batch, hidden_size)
            
        Returns:
            New hidden state of shape (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        # Compute gates
        x_gates = x.data @ self.W_ih.data
        h_gates = h.data @ self.W_hh.data
        
        if self.use_bias:
            x_gates = x_gates + self.bias_ih.data
            h_gates = h_gates + self.bias_hh.data
        
        # Split into individual gates
        x_r = x_gates[:, :self.hidden_size]
        x_z = x_gates[:, self.hidden_size:2*self.hidden_size]
        x_n = x_gates[:, 2*self.hidden_size:]
        
        h_r = h_gates[:, :self.hidden_size]
        h_z = h_gates[:, self.hidden_size:2*self.hidden_size]
        h_n = h_gates[:, 2*self.hidden_size:]
        
        # Apply activations
        r = 1.0 / (1.0 + np.exp(-np.clip(x_r + h_r, -500, 500)))  # Reset gate
        z = 1.0 / (1.0 + np.exp(-np.clip(x_z + h_z, -500, 500)))  # Update gate
        n = np.tanh(x_n + r * h_n)  # New gate
        
        # Update hidden state: h_new = (1 - z) * n + z * h
        h_new_data = (1 - z) * n + z * h.data
        
        # Cache for backward
        self._cache['x'] = x
        self._cache['h'] = h
        self._cache['r'] = r
        self._cache['z'] = z
        self._cache['n'] = n
        self._cache['h_n'] = h_n
        
        h_new = Tensor(
            h_new_data,
            requires_grad=x.requires_grad or h.requires_grad,
            _op='gru_cell',
            _children=(x, h, self.W_ih, self.W_hh)
        )
        
        def _backward(grad):
            r, z, n = self._cache['r'], self._cache['z'], self._cache['n']
            x, h = self._cache['x'], self._cache['h']
            h_n_cached = self._cache['h_n']
            
            # Gradient through hidden state equation
            grad_z = grad * (h.data - n)
            grad_n = grad * (1 - z)
            grad_h_direct = grad * z
            
            # Gradient through new gate (tanh)
            grad_x_n = grad_n * (1 - n ** 2)
            grad_r_h_n = grad_n * (1 - n ** 2)
            
            # Gradient through reset gate
            grad_r = grad_r_h_n * h_n_cached
            grad_h_n = grad_r_h_n * r
            
            # Gradients through sigmoid
            grad_r_pre = grad_r * r * (1 - r)
            grad_z_pre = grad_z * z * (1 - z)
            
            # Concatenate gate gradients for input
            grad_x_gates = np.concatenate([grad_r_pre, grad_z_pre, grad_x_n], axis=1)
            grad_h_gates = np.concatenate([grad_r_pre, grad_z_pre, grad_h_n], axis=1)
            
            # Gradients w.r.t. weights
            if self.W_ih.requires_grad:
                grad_W_ih = x.data.T @ grad_x_gates
                if self.W_ih.grad is None:
                    self.W_ih.grad = Tensor(np.zeros_like(self.W_ih.data))
                self.W_ih.grad.data += grad_W_ih
            
            if self.W_hh.requires_grad:
                grad_W_hh = h.data.T @ grad_h_gates
                if self.W_hh.grad is None:
                    self.W_hh.grad = Tensor(np.zeros_like(self.W_hh.data))
                self.W_hh.grad.data += grad_W_hh
            
            if self.use_bias:
                if self.bias_ih.requires_grad:
                    if self.bias_ih.grad is None:
                        self.bias_ih.grad = Tensor(np.zeros_like(self.bias_ih.data))
                    self.bias_ih.grad.data += np.sum(grad_x_gates, axis=0)
                
                if self.bias_hh.requires_grad:
                    if self.bias_hh.grad is None:
                        self.bias_hh.grad = Tensor(np.zeros_like(self.bias_hh.data))
                    self.bias_hh.grad.data += np.sum(grad_h_gates, axis=0)
            
            # Gradient w.r.t. inputs
            grad_x = grad_x_gates @ self.W_ih.data.T if x.requires_grad else None
            grad_h_prev = grad_h_gates @ self.W_hh.data.T + grad_h_direct if h.requires_grad else None
            
            return (grad_x, grad_h_prev, None, None)
        
        h_new._backward_fn = _backward
        
        return h_new
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = [self.W_ih, self.W_hh]
        if self.use_bias:
            params.extend([self.bias_ih, self.bias_hh])
        return params
    
    def zero_grad(self):
        """Reset gradients."""
        self.W_ih.grad = None
        self.W_hh.grad = None
        if self.use_bias:
            self.bias_ih.grad = None
            self.bias_hh.grad = None
    
    def __repr__(self) -> str:
        return f"GRUCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class GRU:
    """
    Multi-layer GRU.
    
    Processes sequences using stacked GRU cells.
    
    Input shape: (seq_len, batch, input_size)
    Output shape: (seq_len, batch, hidden_size), (num_layers, batch, hidden_size)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_bias: bool = True,
        batch_first: bool = False
    ):
        """
        Initialize GRU.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked GRU layers
            use_bias: Whether to use bias
            batch_first: If True, input shape is (batch, seq, features)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create GRU cells for each layer
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(GRUCell(cell_input_size, hidden_size, use_bias))
    
    def __call__(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all time steps.
        
        Args:
            x: Input tensor of shape (seq_len, batch, input_size) or 
               (batch, seq_len, input_size) if batch_first=True
            h_0: Initial hidden state of shape (num_layers, batch, hidden_size)
            
        Returns:
            outputs: All hidden states (seq_len, batch, hidden_size)
            h_n: Final hidden state (num_layers, batch, hidden_size)
        """
        # Handle batch_first
        if self.batch_first:
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states
        if h_0 is None:
            h = [Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32)) 
                 for _ in range(self.num_layers)]
        else:
            h = [Tensor(h_0.data[i], requires_grad=h_0.requires_grad) 
                 for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                h[i] = cell(layer_input, h[i])
                layer_input = h[i]
            
            outputs.append(h[-1].data)
        
        # Stack outputs
        output_data = np.stack(outputs, axis=0)
        h_n_data = np.stack([h_i.data for h_i in h], axis=0)
        
        if self.batch_first:
            output_data = output_data.transpose(1, 0, 2)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        h_n = Tensor(h_n_data, requires_grad=x.requires_grad)
        
        return output, h_n
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters from all cells."""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all cells."""
        for cell in self.cells:
            cell.zero_grad()
    
    def __repr__(self) -> str:
        return (f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers})")
