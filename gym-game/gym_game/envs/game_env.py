import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import nbimporter
import numpy as np
import math
import sys
if('..' not in sys.path):
    sys.path.insert(0,'..')
if('../..' not in sys.path):
    sys.path.insert(0,'../..')
from Maze.Maze import Maze
from Maze.MazeGenerator import MazeGenerator
from Agents.Worker import Worker
from Main.Simulator import Simulator
from Maths.Cord import Cord
from Maths.Action import Action
from Maths.DQNSolver import DQNSolver

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
       # print("Initialising")
        m= "Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111"
        #"Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111"
        #"Test,6,6,4,5,1,0,131111100051105001100051150001111121" #"Test,10,10,4,0,4,9,1111211151100000000111100110011000000001100001511111011000111000000001100501050110000000011111311111"
        #"Test,4,4,2,0,2,3,1125100110011531" MazeGenerator() 
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.span=6
        self.number=2
        self.pList=[]
        self.stateList=[]
        self.finishedP=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.count=0
        for j in range(self.number):
            p=Worker(self.maze)
            self.s.add(p)
            self.pList.append(p)
            state= np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        action_space=[]
        
        for i in range(0,len(Action)):            
            action_space.append(i)
        self.action_space_worker=np.asarray(action_space)
        self.observation_space_worker= math.pow(2*self.span+1,2)
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        
        self.dqn_solver_worker = DQNSolver(int(self.observation_space_worker), len(self.action_space_worker))
        self.reward=0
        
    def stepAll(self):
        #walls move
        self.history+=str(self.pList[0].getTime())+"#"+ self.maze.returnAllClearString()
        blocked=[]
        for q in self.pList:
            blocked.append(q.getPos())
                        
        self.maze.WallMove(blocked)
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])            
          
        
        #prey move_workers
        index=0
        for p in self.pList:
            state = np.reshape(self.stateList[index],  [1,int(self.observation_space_worker)])
            action = self.dqn_solver_worker.act(state)
            state_next, reward, terminal, info = self.step(p, action, index)
            state_next = np.reshape(state_next, [1,int(self.observation_space_worker)])
            self.dqn_solver_worker.remember(state, action, reward, state_next, terminal)
            self.stateList[index]=state_next
            self.dqn_solver_worker.experience_replay()
            self.reward+=reward
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
            index+=1
        self.history+="|" 
        #predators move
        
        if(self.finished==len(self.pList)):
            file=open("GamesData.txt","a+")
            file.write(self.history+"\n")
            file.close()
            terminal=True  
            
            
        return self.reward, terminal

    def step(self, agent, action, index):  
        stateList=self.stateList
        reward=0
        terminal=False
        info={}
        
        oldPosition=agent.getPos()
        state_next=np.empty(1)      
            
        if(self.action_space_worker[action] in self.maze.WhichWayIsClear(oldPosition, True)):
            agent.Do(self.action_space_worker[action],self.maze)
            state_Next=np.asarray(agent.getView(agent.getPos(),self.span))
            reward+=agent.getReward(agent.getPos(), True,oldPosition,agent.getView(agent.getPos(),self.span))
            self.count+=1            
        else:              
            state_Next=np.asarray(agent.getView(agent.getPos(),self.span))                
            reward+=agent.getReward(agent.getPos(), False,oldPosition,agent.getView(agent.getPos(),self.span))
            
        if(self.maze.CheckExit(agent.getPos()) and agent not in self.finishedP):
            self.finished+=1
            self.finishedP.append(agent)           
            
        
        return state_Next, reward, terminal, info
    
    def reset(self):        
        #print("Resetting")
        self.maze= Maze(self.maze.mazeString)
        self.stateList=[]
        self.history=self.maze.mazeString+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.count=0
        for p in self.pList:
            p.setInitPos(Cord(self.maze.getInitialX(),self.maze.getInitialY()))
            state=np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        #print(self.maze.mazeString)
        #self.maze.printMaze()
        return self.stateList
    
    def resetNewMaze(self):        
        m= MazeGenerator()  
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.pList=[]
        self.stateList=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.count=0
        for j in range(self.number):
            p=Worker(self.maze)
            self.pList.append(p)
            self.s.add(p)
            state= np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        return self.stateList
        
    def render(self, mode='human', close=False):
        self.s.display()