import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ml_agent import RubiksCFOPNet, MOVES

def state_list_to_tensor(state_list):
    """Converts a list of 54 numeric strings (0-5) into a 324-element one-hot tensor."""
    tensor = torch.zeros(54 * 6)
    for i, val in enumerate(state_list):
        try:
            color_idx = int(val)
            if 0 <= color_idx <= 5:
                tensor[i * 6 + color_idx] = 1.0
        except ValueError:
            pass
    return tensor

def move_to_index(move_str):
    """Maps a move notation (e.g., "U'") to its integer index."""
    return MOVES.index(move_str) if move_str in MOVES else 0

class RubiksDataset(Dataset):
    def __init__(self, data_dir, max_files=5):
        self.X = []
        self.Y = []
        
        # Target the .seq files as they map 1 State -> 1 Next Move directly
        file_pattern = os.path.join(data_dir, "training.seq.[0-9]*")
        files = glob.glob(file_pattern)
        
        files = sorted(files)[:max_files]
        print(f"Found {len(files)} sequence files to process...")

        for file_path in files:
            print(f"Parsing {os.path.basename(file_path)}...")
            with open(file_path, 'r') as f:
                current_state = None
                
                for line in f:
                    parts = line.strip().split()
                    
                    # If we see exactly 54 items, it's a cube state line
                    if len(parts) == 54:
                        current_state = parts
                        
                    # If we see exactly 1 item and we have a state buffered, it's the move line
                    elif len(parts) == 1 and current_state is not None:
                        move = parts[0]
                        
                        # '#' denotes the end of a sequence, so we don't train on it
                        if move != '#':
                            self.X.append(state_list_to_tensor(current_state))
                            self.Y.append(move_to_index(move))
                            
                        # Reset the buffer for the next pair
                        current_state = None

        if len(self.X) == 0:
            raise ValueError("No valid data was parsed. Check the dataset path and format.")

        self.X = torch.stack(self.X)
        self.Y = torch.tensor(self.Y, dtype=torch.long)
        print(f"Successfully loaded {len(self.X)} state-action pairs.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train():
    # 1. Setup Data
    dataset_dir = os.path.join("archive", "dataset")
    # Increase max_files if you want to train on more data, but 2 is good for a quick test
    dataset = RubiksDataset(dataset_dir, max_files=2) 
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 2. Setup Model
    model = RubiksCFOPNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 5
    print("\nStarting Training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 4. Save Weights
    torch.save(model.state_dict(), "rubiks_cfop_weights.pth")
    print("\nTraining complete! Model saved as rubiks_cfop_weights.pth")

if __name__ == "__main__":
    train()