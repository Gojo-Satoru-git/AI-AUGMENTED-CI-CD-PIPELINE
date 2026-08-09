import torch
import torch.nn as nn

class TraceAnomalyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(TraceAnomalyLSTM, self).__init__()
        
        # 1. Embedding Layer
        # padding_idx=0 tells PyTorch to ignore all those 0s in your sequences!
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. LSTM Layer
        # batch_first=True tells PyTorch our data is formatted as (Batch, Sequence, Features)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        # 3. Fully Connected (Linear) Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 4. Sigmoid Activation (Outputs a probability between 0 and 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> e.g., (64, 100)
        
        # Step A: Convert integers to vectors
        embedded = self.embedding(x)
        
        # Step B: Pass through LSTM
        # 'hidden' contains the final memory state of the entire sequence
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Step C: Grab the memory from the very last LSTM layer
        final_hidden_state = hidden[-1]
        
        # Step D: Pass through the linear layer and squash to a probability
        out = self.fc(final_hidden_state)
        return self.sigmoid(out).squeeze()