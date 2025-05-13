import environment
from DQN_CNN import DQN
import random
import torch
path = "Data/parameters1"
class DQN_Agent:
    def __init__(self,parameters_path =None,train =False,env =None):
        self.parameters_path=parameters_path
        self.train=train
        self.env=env
        self.DQN : DQN = DQN()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def getAction(self, state, epoch = 0, events= None, train =True): 
        actions = [0,1,2,3]
        if train:
            epsilon = self.DQN.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        with torch.no_grad():
            Q_values = self.DQN(state.unsqueeze(0))
        max_index = torch.argmax(Q_values)
        return actions[max_index]
        # return random.choice(actions)
    
    def get_max_action_value(self, states): 
        
        with torch.no_grad():
            Q_values = self.DQN(states)
        
        max_Q_values, actions = torch.max(Q_values, dim=1)

        max_Q_values = max_Q_values.unsqueeze(1)   # [B, 1]
        actions = actions.unsqueeze(1) 
        return max_Q_values, actions
        
    

    def get_action_values(self, states, actions):
        """
        states: Tensor of shape [B, 1, 9, 9]
        actions: Tensor of shape [B, 1] (int64)
        returns: Tensor of shape [B] - the Q-value for each (state, action)
        """

        # Forward pass through the network to get Q-values for all actions
        q_values = self.DQN(states)  # output shape: [B, 4]

        # Make sure actions are in correct shape [B]
        actions = actions.squeeze(1).to(self.DQN.device)  # [B]

        # Select the Q-value corresponding to each action
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        return selected_q_values.unsqueeze(1) #[B, 1]

    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)
