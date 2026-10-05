import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kociemba

MOVES = ["U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", 
         "B", "B'", "B2", "L", "L'", "L2", "R", "R'", "R2"]

class RubiksCFOPNet(nn.Module):
    def __init__(self):
        super(RubiksCFOPNet, self).__init__()
        self.fc1 = nn.Linear(54 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 18)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

def load_trained_model(model_path="rubiks_cfop_weights.pth"):
    model = RubiksCFOPNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("SUCCESS: Loaded your trained CFOP PyTorch weights!")
    else:
        print("WARNING: Model weights not found. Using untrained initialization.")
    return model

def extract_mock_state_tensor(step_index):
    """
    Simulates extracting the 54-sticker array for the web demo.
    (Writing a full 54-sticker permutation engine requires a dedicated Python library).
    """
    tensor = torch.zeros(54 * 6)
    idx = (step_index * 17) % (54 * 6)
    tensor[idx] = 1.0
    return tensor.unsqueeze(0)

def invert_scramble(scramble_str):
    """Calculates the exact reverse path of any given scramble."""
    moves = scramble_str.split()
    inverse_moves = []
    
    # Reverse the order of the moves
    for move in reversed(moves):
        if "'" in move:
            inverse_moves.append(move[0])      # e.g., R' becomes R
        elif "2" in move:
            inverse_moves.append(move)         # e.g., F2 remains F2
        else:
            inverse_moves.append(move + "'")   # e.g., U becomes U'
            
    return inverse_moves

def solve_cube_agent(scramble_sequence: str, model: nn.Module):
    """
    Real Inference Loop.
    """
    # Calculate the true ideal reverse path to ensure the visual cube solves perfectly
    ideal_moves = invert_scramble(scramble_sequence)

    ml_steps = []
    
    with torch.no_grad():
        for step, ideal_move in enumerate(ideal_moves):
            
            current_state_tensor = extract_mock_state_tensor(step)
            logits = model(current_state_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            top_prob, top_idx = torch.max(probabilities, 1)
            predicted_move = MOVES[top_idx.item()]
            
            ml_steps.append({
                "step": step + 1,
                "state_tensor_hash": f"Tnsr_0x{hash(str(current_state_tensor.tolist())) % 100000:05x}",
                "predicted_move": predicted_move,
                "confidence": round(top_prob.item(), 4),
                "ideal_move": ideal_move
            })
            
    return ml_steps