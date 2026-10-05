import numpy as np
import gymnasium as gym
from gymnasium import spaces

MOVES = ["U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", 
         "B", "B'", "B2", "L", "L'", "L2", "R", "R'", "R2"]

class RubiksEnv(gym.Env):
    """
    Physically accurate 3x3 Rubik's Cube Environment.
    Tracks all 54 facelets.
    0-8: Up, 9-17: Right, 18-26: Front, 27-35: Down, 36-44: Left, 45-53: Back
    """
    def __init__(self):
        super(RubiksEnv, self).__init__()
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=0, high=1, shape=(324,), dtype=np.float32)
        
        # A solved cube: 9 of each color (0-5) grouped together
        self.solved_state = np.repeat(np.arange(6), 9)
        self.state = np.copy(self.solved_state)
        self.step_count = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = np.copy(self.solved_state)
        
        # Scramble with 20 physically accurate random moves
        for _ in range(20):
            action = np.random.randint(0, 18)
            self._apply_move(action)
            
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # Apply the exact physical twist
        self._apply_move(action)
        
        obs = self._get_obs()
        matches = np.sum(self.state == self.solved_state)
        
        if matches == 54:
            reward = 1000.0
            terminated = True
        else:
            # Reward shaping: Positive reinforcement for getting closer, small penalty for time
            reward = (matches * 0.1) - 1.0
            terminated = False
            
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Converts the 54-item array into a 324-dim PyTorch tensor."""
        obs = np.zeros((54, 6), dtype=np.float32)
        obs[np.arange(54), self.state] = 1.0
        return obs.flatten()

    def _apply_move(self, action_idx):
        """Translates the 18 move variations into base mathematical rotations."""
        move_name = MOVES[action_idx]
        base_move = move_name[0]
        
        # Determine rotation count (e.g., U' is 3 clockwise rotations, U2 is 2)
        if "'" in move_name:
            repeats = 3
        elif "2" in move_name:
            repeats = 2
        else:
            repeats = 1
            
        for _ in range(repeats):
            self._rotate_base_face(base_move)

    def _rotate_base_face(self, face):
        """Executes the exact 20-sticker array permutation for a 90-degree clockwise twist."""
        s = self.state
        
        if face == 'U':
            s[0:9] = [s[6], s[3], s[0], s[7], s[4], s[1], s[8], s[5], s[2]]
            s[18:21], s[36:39], s[45:48], s[9:12] = s[9:12].copy(), s[18:21].copy(), s[36:39].copy(), s[45:48].copy()
            
        elif face == 'D':
            s[27:36] = [s[33], s[30], s[27], s[34], s[31], s[28], s[35], s[32], s[29]]
            s[24:27], s[15:18], s[51:54], s[42:45] = s[42:45].copy(), s[24:27].copy(), s[15:18].copy(), s[51:54].copy()

        elif face == 'F':
            s[18:27] = [s[24], s[21], s[18], s[25], s[22], s[19], s[26], s[23], s[20]]
            temp = s[6:9].copy()
            s[6], s[7], s[8] = s[44], s[41], s[38]
            s[44], s[41], s[38] = s[29], s[28], s[27]
            s[29], s[28], s[27] = s[15], s[12], s[9]
            s[15], s[12], s[9] = temp[2], temp[1], temp[0]

        elif face == 'B':
            s[45:54] = [s[51], s[48], s[45], s[52], s[49], s[46], s[53], s[50], s[47]]
            temp = s[0:3].copy()
            s[0], s[1], s[2] = s[17], s[14], s[11]
            s[17], s[14], s[11] = s[33], s[34], s[35]
            s[33], s[34], s[35] = s[36], s[39], s[42]
            s[36], s[39], s[42] = temp[2], temp[1], temp[0]

        elif face == 'L':
            s[36:45] = [s[42], s[39], s[36], s[43], s[40], s[37], s[44], s[41], s[38]]
            temp = [s[0], s[3], s[6]]
            s[0], s[3], s[6] = s[53], s[50], s[47]
            s[53], s[50], s[47] = s[33], s[30], s[27]
            s[33], s[30], s[27] = s[24], s[21], s[18]
            s[24], s[21], s[18] = temp[2], temp[1], temp[0]

        elif face == 'R':
            s[9:18] = [s[15], s[12], s[9], s[16], s[13], s[10], s[17], s[14], s[11]]
            temp = [s[2], s[5], s[8]]
            s[2], s[5], s[8] = s[20], s[23], s[26]
            s[20], s[23], s[26] = s[29], s[32], s[35]
            s[29], s[32], s[35] = s[51], s[48], s[45]
            s[51], s[48], s[45] = temp[2], temp[1], temp[0]