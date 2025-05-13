import pygame
import torch
from environment import Game
from DQN_Agent import DQN_Agent
import graphics
import os
from ReplayBuffer import ReplayBuffer
import wandb

WIDTH , HEIGHT = 540,710
def main (chkpt):

    pygame.init()
    path = "Data/parameters2"
    bufferPath = "Data/buffer2"
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption('PAC MAN')
    game = Game()

    best_score = 0

    #region ###### params ############
    game = Game()
    player = DQN_Agent(path, env=game)
    # player.load_params(path)
    # player.save_param(path)
    player_hat = DQN_Agent(env=game)
    player_hat.DQN = player.DQN.copy()
    batch_size = 32
    buffer = ReplayBuffer()
    learning_rate = 1e-4
    epochs = 100000
    start_epoch = 0
    gamma=0.5
    C = 5
    loss = torch.tensor(-1,dtype=torch.float32,requires_grad=True)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[1000*200, 2000*200, 5000*2000], gamma=gamma)
    step = 0
    #endregion
    #region   ############# wandb init ###########################
    wandb.init(
    # set the wandb project where this run will be logged
        project="pacman-project",
        id=f'PacMan{chkpt}',
        name=f"PacMan{chkpt}",
        config={
        "learning_rate": learning_rate,
        "architecture": "DQN",
        "dataset": "PACMAN",
        "epochs": epochs,
        "batch_size":batch_size,
        "gamma":gamma

        }
    )
    #endregion
    #region ######## checkpoint Load ############
    checkpoint_path = "Data/checkpoint1.pth"
    buffer_path = "Data/buffer1.pth"
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     start_epoch = checkpoint['epoch']+1
    #     player.DQN.load_state_dict(checkpoint['model_state_dict'])
    #     player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
    #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     buffer = torch.load(buffer_path)
    #     losses = checkpoint['loss']
    #     scores = checkpoint['scores']
    #     avg_score = checkpoint['avg_score']
    # player.DQN.train()
    # player_hat.DQN.eval()

    #endregion################################

    for epoch in range(start_epoch, epochs):
        game.reset()
        state_cnn = game.state_cnn()
        gameTick=0
        run = True
        steps=0
        while run:
            pygame.event.pump()
            events = pygame.event.get()
            for event in events:
                if event.type==pygame.QUIT:
                    run=False
            gameTick=gameTick%264
            ############## Sample Environement #########################
            if gameTick%6==0:
                step+=1
                action = player.getAction(state=state_cnn, epoch=epoch,train=True)
                gameTick,nextState,reward=game.tick(gameTick, action)
                buffer.push(state_cnn, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                            nextState, torch.tensor(game.game_over!=False, dtype=torch.float32))
                graphics.Graphics.game_screen(screen,game)
                pygame.display.update()
            else:
                gameTick,next_state,_reward=game.tick(gameTick,action)
                reward += _reward
                graphics.Graphics.game_screen(screen,game)
                pygame.display.update()

            if game.game_over or step > 300:
                best_score = max(best_score, game.points)
                if game.game_over == 'lose':
                    reward = game.lose_reward
                elif game.game_over == 'win':
                    reward = game.win_reward
                buffer.push(state_cnn, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                            next_state, torch.tensor(game.game_over=='lose', dtype=torch.float32))
                graphics.Graphics.game_screen(screen,game)
                pygame.display.update()
                break
            state_cnn = nextState

            pygame.display.update()
            clock.tick(1000)
            
            if len(buffer) < 100:
                continue
    
            ############## Train ################
            if gameTick%6==0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)  
                Q_values = player.get_action_values(states, actions)
                
                Q_hat_Values, next_actions = player_hat.get_max_action_value(next_states)    # DQN
                
                loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)   # loss = (r + q_hat(s',a') - q(s,a))^2

                loss.backward()
                optim.step()
                optim.zero_grad()
                scheduler.step()
                
        # print (f"epoch: {epoch}, step:{step}, score: {game.points}, best score: {best_score}, loss: {loss}, reward:{reward}", end="\r")
                
        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())
            # player.save_param(path)
        
        
       
        #region ########### log and print #############################

        
        # torch.save(buffer,bufferPath)
        

        print (f'chkpt:{chkpt} epoch: {epoch} loss: {loss:.7f} LR: {scheduler.get_last_lr()} step: {step} ' \
               f'score: {game.points} best_score: {best_score}')
        wnblog = {'step': {step}, 'loss': loss,'score':game.points}
        wandb.log(wnblog)
        
        step = 0
        scores.append(game.points)
        losses.append(loss.item())

        avg = (avg * (epoch % 10) + game.points) / (epoch % 10 + 1)
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            print (f'average score last 10 games: {avg} ')
            avg = 0

        
        
        # if epoch % 5 == 0 and epoch > 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': player.DQN.state_dict(),
        #         'optimizer_state_dict': optim.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': losses,
        #         'scores':scores,
        #         'avg_score': avg_score
        #     }
            
            # torch.save(checkpoint, checkpoint_path)
            # torch.save(buffer, buffer_path)
           
        #endregion

        


        
if __name__ == "__main__":
    if not os.path.exists("Data/checkpoit_num"):
        torch.save(1, "Data/checkpoit_num")    
    
    chkpt = torch.load("Data/checkpoit_num", weights_only=False)
    chkpt += 1
    torch.save(chkpt, "Data/checkpoit_num")    
    main (chkpt)