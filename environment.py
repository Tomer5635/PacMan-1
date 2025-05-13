import numpy as np
import math
import random
import torch
import torch.nn.functional as F

class Game:
    def __init__(self):

        self.ghostHomeTiles = [(1,26),(1,1),(29,26),(29,1)]
        self.board = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 5,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 5,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0,10, 0, 0, 0, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-2,-2,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 0, 7, 0, 8, 0, 9,-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 5, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,11, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 5,-1],
                [-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1],
                [-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]
        self.board = np.array(self.board,dtype=int)
        self.direction = 0
        self.nextDirection = 0
        self.points = 0
        self.ghostModes = [1,1,1,1] #0=chase 1=scatter 2=frightened
        self.ghostDirections = [0,3,3,3]
        self.ghostUnder = np.array([0,0,-3,0],dtype=int)
        self.ghostTargetTiles = [(1,26),(1,1),(29,26),(29,1)]
        self.waveCounter=0
        self.waveTime=0
        self.frightenedTime=0
        self.game_over = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_rewards()
        self.prev_action = 0

    def init_pacman (self):
        default =(23, 14)
        locations = [default, (1, 10), (1, 17), (5, 7), (5, 15), (29,8), (29,22), (26, 5), (26, 12), (26, 26), (23, 14)]
        new_locatiom = random.choice(locations)
        self.board[default] = 1
        self.board[new_locatiom] = 11

    def init_rewards (self):
        self.reward = 0
        self.food_reward = 3
        self.ball_reward = 10
        self.step_reward = -0.3
        self.ghost_eat_reward = 20
        self.lose_reward = -50
        self.win_reward = 50
        self.reverse_reward = 0

    def move(self):
        if 11 in self.board:
            i,j=np.argwhere(self.board == 11)[0]
        if self.board[i][j] ==11:
            if self.nextDirection==0:
                if self.board[i][(j+1)%28]>=0:
                    self.direction=0
            elif self.nextDirection==1:
                if self.board[i+1][j]>=0:
                    self.direction=1
            elif self.nextDirection==2:
                if self.board[i][j-1]>=0:
                    self.direction=2
            elif self.nextDirection==3:
                if self.board[i-1][j]>=0:
                    self.direction=3
            if self.direction==0:
                nextPos = (i,(j+1)%28)
            elif self.direction==1:
                nextPos = (i+1,j)
            elif self.direction==2:
                nextPos = (i,j-1)
            elif self.direction==3:
                nextPos = (i-1,j)
            if self.board[nextPos[0]][nextPos[1]]!=-1 and self.board[nextPos[0]][nextPos[1]]!=-2:
                if self.board[nextPos[0]][nextPos[1]]>6:
                    self.reward += self.lose_reward
                    self.gameOver()
                    return
                if self.board[nextPos[0]][nextPos[1]]<-6:
                    self.respawnGhost(self.board[nextPos[0]][nextPos[1]]+10,nextPos)
                self.board[i][j] = 0
                if self.board[nextPos[0]][nextPos[1]]==1:
                    self.points+=10
                    self.reward += self.food_reward
                elif self.board[nextPos[0]][nextPos[1]]==5:
                    self.points+=50
                    self.reward +=self.ball_reward
                    self.ghostModes=[2,2,2,2]
                    self.frightenedTime=0
                else:
                    self.reward += self.step_reward
                self.board[nextPos[0]][nextPos[1]]=11


    def ghostModeUpdate(self):
        if self.waveCounter%2==0:
            mode=1
        else:
            mode=0
        if 2 not in self.ghostModes:
            self.waveTime+=1
            self.frightenedTime=0
            if self.waveCounter>6:
                self.waveTime=-999999
            elif self.waveCounter%2==1:
                if self.waveTime==20:
                    self.waveCounter+=1
                    self.waveTime=0
                    self.ghostModes=[1,1,1,1]
            else:
                if self.waveCounter>3:
                    if self.waveTime==5:
                        self.waveCounter+=1
                        self.waveTime=0
                        self.ghostModes=[0,0,0,0]
                else:
                    if self.waveTime==7:
                        self.waveCounter+=1
                        self.waveTime=0
                        self.ghostModes=[0,0,0,0]
        else:
            self.frightenedTime+=1
            if self.frightenedTime==7:
                self.frightenedTime=0
                for i in range(min(2+int(self.points/300),4)):
                    if self.ghostModes[i] ==2:
                        self.ghostModes[i]=mode
                        coords = np.argwhere(self.board == i - 10)
                        if coords.size > 0:
                            x,y=np.argwhere(self.board==i-10)[0]
                            self.board[x][y]=self.board[x][y]*-1
    
    def ghostMove(self,ghost):
        if ghost <4-int(self.points<300)-int(self.points<600):
            if 10-ghost not in self.board:
                self.ghostMove(ghost+1)
                return
            i,j=np.argwhere(self.board==10-ghost)[0]
            self.determineTargetTile(ghost)
            options = []
            if (self.board[i-1][j]>=0   or (self.board[i-1][j]==-2 and i==13) or self.board[i-1][j]==-10+ghost) and self.ghostDirections[ghost]!=1:
                options.insert(0,(i-1,j))
            if (self.board[i+1][j]>=0  or self.board[i+1][j]==ghost-10) and self.ghostDirections[ghost]!=3:
                options.insert(0,(i+1,j))
            if (self.board[i][(j-1)%28]>=0 or self.board[i][(j-1)%28]==ghost-10) and self.ghostDirections[ghost]!=0:
                options.insert(1,(i,(j-1)%28))
            if (self.board[i][(j+1)%28]>=0 or self.board[i][(j+1)%28]==ghost-10) and self.ghostDirections[ghost]!=2:
                options.insert(0,(i,(j+1)%28))
            if(len(options)==0):
                self.ghostMove(ghost+1)
                return False
            
            least = options[0]
            if len(options)>1:
                for index in range(len(options)):
                    if math.hypot(options[index][0]-self.ghostTargetTiles[ghost][0],options[index][1]-self.ghostTargetTiles[ghost][1])<=math.hypot(least[0]-self.ghostTargetTiles[ghost][0],least[1]-self.ghostTargetTiles[ghost][1]):
                        least=options[index]
            nextPos=least
            if j+1==nextPos[1]:
                dir=0
            elif j-1==nextPos[1]:
                    dir=2
            elif i-1==nextPos[0]:
                dir=3
            else:
                dir=1
            if self.board[nextPos[0]][nextPos[1]]==11:
                self.gameOver()
                return False
            self.ghostDirections[ghost]=dir
            self.board[i][j]=self.ghostUnder[ghost]
            self.ghostUnder[ghost]=self.board[nextPos[0]][nextPos[1]]
            if self.ghostModes[ghost]==2:
                self.board[nextPos[0]][nextPos[1]] = ghost-10
            else:
                self.board[nextPos[0]][nextPos[1]] = 10-ghost
            self.ghostMove(ghost+1)
            return True

    def blueGhostMove(self,ghost):
        if ghost <4-int(self.points<300)-int(self.points<600):
            if ghost-10 not in self.board:
                self.blueGhostMove(ghost+1)
                return True
            i,j=np.argwhere(self.board==ghost-10)[0]
            options = []
            if (self.board[i-1][j]>=0 or (self.board[i-1][j]==-2 and i==13) or self.board[i-1][j]==-10+ghost) and self.ghostDirections[ghost]!=1:
                options.insert(0,(i-1,j))
            if (self.board[i+1][j]>=0 or self.board[i+1][j]==ghost-10) and self.ghostDirections[ghost]!=3:
                options.insert(0,(i+1,j))
            if (self.board[i][j-1]>=0 or self.board[i][j-1]==ghost-10) and self.ghostDirections[ghost]!=0:
                options.insert(1,(i,(j-1)%28))
            if (self.board[i][(j+1)%28]>=0 or self.board[i][(j+1)%28]==ghost-10) and self.ghostDirections[ghost]!=2:
                options.insert(0,(i,(j+1)%28))
            if(len(options)==0):
                self.ghostDirections[ghost]=(self.ghostDirections[ghost]+2)%4
                self.blueGhostMove(ghost+1)
                return
            nextPos=options[random.randint(0,len(options)-1)]
                
            if j+1==nextPos[1]:
                dir=0
            elif j-1==nextPos[1]:
                    dir=2
            elif i-1==nextPos[0]:
                dir=3
            else:
                dir=1
            if self.board[nextPos[0]][nextPos[1]]==11:
                self.respawnGhost(ghost,(i,j))
                self.blueGhostMove(ghost+1)
                return
            self.ghostDirections[ghost]=dir
            self.board[i][j]=self.ghostUnder[ghost]
            self.ghostUnder[ghost]=self.board[nextPos[0]][nextPos[1]]
            self.board[nextPos[0]][nextPos[1]] = ghost-10
            self.blueGhostMove(ghost+1)
            return True

    def determineTargetTile(self,ghost):
        coords = np.argwhere(abs(self.board)==10-ghost)[0]
        if (coords[0] >=13 and coords[0]<=15) and coords[1]>10 and coords[1]<16:
            self.ghostTargetTiles[ghost]=(11,14)
            return
        if self.ghostModes[ghost] == 1:
            self.ghostTargetTiles[ghost]=self.ghostHomeTiles[ghost]
        elif self.ghostModes[ghost]==0:
            pacman=np.argwhere(self.board==11)[0]
            if ghost==0:
                self.ghostTargetTiles[0]= pacman
            if ghost==1:
                self.ghostTargetTiles[1]=pacman
                if self.direction == 0:
                    self.ghostTargetTiles[1]=(self.ghostTargetTiles[1][0],self.ghostTargetTiles[1][1]+4)
                if self.direction == 1:
                    self.ghostTargetTiles[1]=(self.ghostTargetTiles[1][0]+4,self.ghostTargetTiles[1][1])
                if self.direction == 2:
                    self.ghostTargetTiles[1]=(self.ghostTargetTiles[1][0],self.ghostTargetTiles[1][1]-4)
                if self.direction == 3:
                    self.ghostTargetTiles[1]=(self.ghostTargetTiles[1][0]-4,self.ghostTargetTiles[1][1])
            if ghost==2:
                if 10 in self.board or -10 in self.board:
                    blinky=np.argwhere(abs(self.board)==10)[0]
                else:
                    blinky=np.argwhere(abs(self.board)==10-np.argwhere(abs(self.ghostUnder)==10)[0])[0]
                halfway=pacman
                if self.direction == 0:
                    halfway=(halfway[0],halfway[1]+2)
                if self.direction == 1:
                    halfway=(halfway[0]+2,halfway[1])
                if self.direction == 2:
                    halfway=(halfway[0],halfway[1]-2)
                if self.direction == 3:
                    halfway=(halfway[0]-2,halfway[1])
                self.ghostTargetTiles[2]=(2*halfway[0]-blinky[0],2*halfway[1]-blinky[1])
            if ghost==3:
                clyde=np.argwhere(self.board==7)[0]
                if math.hypot(pacman[0]-clyde[0],pacman[1]-clyde[1])>8:
                    self.ghostTargetTiles[3]= pacman
                else:
                    self.ghostTargetTiles[3]=self.ghostHomeTiles[3]

    def respawnGhost(self,ghost,pos):
        self.board[13,14] = 10-ghost
        self.points+=200
        self.reward += self.ghost_eat_reward
        self.ghostDirections[ghost]=3
        if self.waveCounter%2==0:
            self.ghostModes[ghost]=1
        else:
            self.ghostModes[ghost]=0
        self.board[pos]=self.ghostUnder[ghost]
        self.ghostUnder[ghost]=0

    def gameOver(self):
        print("GAME OVER")
        self.game_over="lose"

    def win(self):
        self.game_over="win"

    def getLegalActions(self):
        return [0,1,2,3]

    def tick(self,GameTick,action):
        if not self.game_over:
            prevPoints = self.points
            if 11 not in self.board:
                self.gameOver()
                self.reward += self.lose_reward
            if (1 not in self.board and 5 not in self.board) and (1 not in self.ghostUnder and 5 not in self.ghostUnder):
                self.win()
                self.reward += self.win_reward
            
            if GameTick%6==0:
                if action is not None:
                    # if abs (self.nextDirection - action) == 2:
                    #     self.reward =+ self.reverse_reward
                    self.nextDirection = action
                    
                self.move()
            if GameTick%8==0:
                self.ghostMove(0)
            if GameTick%11==0:
                self.blueGhostMove(0)
            if GameTick%60==0:
                self.ghostModeUpdate()
            reward = self.reward
            self.reward = 0
            # reward=(self.points-prevPoints)/10
            return GameTick+1,self.state_cnn(),reward        ##################
        return GameTick

    def state(self):
        board_np = self.board.reshape(868) # 31 * 28
        p_dir = self.direction
        g_dir = self.ghostDirections.copy()
        g_dir.append(p_dir)
        directions = np.array(g_dir,dtype=np.float32)
        state=np.concatenate([board_np,directions])
        return torch.tensor(state,dtype=torch.float32)
        
    def state_cnn(self):
        board = np.zeros_like(self.board, dtype=float)
        board[(self.board==-1) | (self.board==-2)] = -0.1  # wall -1, -2 -> -1
        board[self.board==0] = 0  # empty 0 -> 0
        board[self.board==1] = 0.1  # food 1 -> 1
        board[self.board==5] = 0.2  # ball 5 -> 2
        board[(self.board >=7) &  (self.board <= 10) ] = -0.5 # bad_ghost
        board[(self.board>= -10) & (self.board <= -7)] = 0.5 # eatable ghost
        board[self.board==11] = 1  # pac man 11 -> 10        
        
        # pacman = np.zeros_like(self.board)
        # pacman[self.board==11] = self.direction + 10

        # ghost = np.zeros_like(self.board)
        # ghost[(self.board >=7) &  (self.board <= 10) ] = -5 # bad_ghost
        # ghost[(self.board>= -10) & (self.board <= -7)] = 5 # eatable ghost
        
        # ghost_eatable = np.zeros_like(self.board) 
        # ghost_eatable[(self.board>= -10) & (self.board <= -7)] = 1 # add the direction - next tiel with 0.1
        # state = np.stack([board, pacman, ghost], axis=0)  
        state = torch.tensor(board, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0)
        return self.get_sub_state(state)
        
        
    def state_actions (self, state_cnn):
        actions, next_pacman_lst = self.legal_actions(state_cnn)
        batch = []
        for next_pacman in next_pacman_lst:
            state = torch.stack([state_cnn[0], next_pacman, state_cnn[2]], axis=0) 
            batch.append(state)
        
        batch_state = torch.stack(batch, axis=0)
        return batch_state, actions

    def legal_actions(self, state_cnn):
        board = state_cnn[0]
        pacman = state_cnn[1]
        rows, cols = pacman.shape
        row, col = torch.where(pacman >= 10)
        row = row.item()
        col = col.item()
        legal_actions = []
        next_pacman_lst = []
        for action in range(4):
            if action == 0 and col + 1 < cols and board[row, col+1] != -1:
                next_pacman = torch.zeros_like(pacman)
                next_pacman[row, col+1] = 10 + self.direction
                legal_actions.append(action)
                next_pacman_lst.append(next_pacman)
            elif action == 1 and row + 1 < rows and board[row+1, col] != -1:
                next_pacman = torch.zeros_like(pacman)
                next_pacman[row+1, col] = 10 + self.direction
                legal_actions.append(action)
                next_pacman_lst.append(next_pacman)
            elif action == 2 and col - 1 >= 0 and board[row, col-1] != -1:
                next_pacman = torch.zeros_like(pacman)
                next_pacman[row, col-1] = 10 + self.direction
                legal_actions.append(action)
                next_pacman_lst.append(next_pacman)
            elif action == 3 and row - 1 >= 0 and board[row-1, col] != -1:
                next_pacman = torch.zeros_like(pacman)
                next_pacman[row-1, col] = 10 + self.direction
                legal_actions.append(action)
                next_pacman_lst.append(next_pacman)

        return legal_actions, next_pacman_lst

    def get_sub_state (self, state_cnn, n=4):
        row, col = torch.where(state_cnn.squeeze() == 1) # pac man position
        row = row.item() + n
        col = col.item() + n
        pad = (n, n, n, n)

        state_cnn = state_cnn.unsqueeze(0) 

        # Pad the board
        padded = F.pad(state_cnn, pad, mode='circular')
        
        padded = padded.squeeze(0)
        
        sub_state = padded[:, row-n:row+n+1, col-n:col+n+1]

        return sub_state

                
    def reset (self):
        self.ghostHomeTiles = [(1,26),(1,1),(29,26),(29,1)]
        self.board = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 5,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 5,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0,10, 0, 0, 0, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-2,-2,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 0, 7, 0, 8, 0, 9,-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1, 0, 0, 0, 0, 0],
                [-1,-1,-1,-1,-1,-1, 1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1],
                [-1, 5, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,11, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 5,-1],
                [-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1],
                [-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
                [-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]
        self.board = np.array(self.board,dtype=int)
        self.direction = 0
        self.nextDirection = 0
        self.points = 0
        self.ghostModes = [1,1,1,1] #0=chase 1=scatter 2=frightened
        self.ghostDirections = [0,3,3,3]
        self.ghostUnder = np.array([0,0,-3,0],dtype=int)
        self.ghostTargetTiles = [(1,26),(1,1),(29,26),(29,1)]
        self.waveCounter=0
        self.waveTime=0
        self.frightenedTime=0
        self.game_over = False
        self.reward = 0
        self.init_pacman()
    
                
                            
        
                       
    
                
                
                        
        
                
        
            
                    
        


