import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import pandas as pd
import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    #create global 'actions' list
    global actions 
    actions = [None, 'forward', 'left', 'right']
    #create global dictionary to store Q values
    global Q
    #use defaultdict to initialize keys in Q, when used for the first time 
    Q = defaultdict(dict)

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        #create 'successes' list.Used to store successes as well as failures, however, as for each run the total number of routes will be 100
        #success just stores the times the goal is reached
        #could/should be changed to an integer instead
        self.success = 0
        self.successes = []
        #initialize action and old state to None
        self.action = None
        self.old_state = None
        #count the number of runs (used to reduce the epsilon with time)
        self.runs = 0
        self.rewards = []
        self.totalRewards = []
        self.rewardsPerStep = []
        self.rewardsTable = []
        self.moves = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        #self.runs will not be reset, so that epsilon is reduced continuously while alpha and gamma and initial epsilon are the same
        self.moves.append(self.runs)
        self.totalRewards.append(np.sum(self.rewards))
        if self.runs > 0:
            self.rewardsPerStep.append(np.sum(self.rewards)/self.runs)
        self.rewardsTable.append(self.rewards)
        self.runs = 0
        self.successes.append(self.success)
        self.success = 0
        self.rewards = []
        
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        
        #update runs number
        self.runs +=1
        
        #adjust epsilon according to the number of runs
        self.epsilon_adjusted = self.epsilon/self.runs
        
        # TODO: Update state
        self.state = {inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint}
        #since all the items that are None are grouped into one None, None in the state is discarted
        self.state.discard(None)
        #sorting the state elements, so when they are turned into string and inserted as keys into Q, we don't have the same states multiple times
        self.state = sorted(self.state)

        #if the current state is not in Q, give all the random initial values between -0.5 and 0.5 to all actions
        if ({str(self.state)}.issubset(Q)):
            pass
        else:
            dict1 = {}
            for act in actions:
                dict1[str(act)] =  random.random() - 0.5
            Q[str(self.state)] = dict1

            
       #choose a random number, and if it is less than epsilon, choose a random action
        #otherwise, choose action with the highest Q value
        if random.random()<self.epsilon_adjusted or self.old_state == None:
            self.action = actions[int(4*random.random())]
        else:
            act_vals = []
            for act in actions:
                act_vals.append(Q[str(self.state)][str(act)])
            self.action = actions[act_vals.index(max(act_vals))]
        #if the chosen action is None, we have to change it, since it cannot be a string
        if self.action == 'None':
            self.action = None
        # Execute action and get reward
        reward = self.env.act(self, self.action)
        
        #add it to the rewards table
        self.rewards.append(reward)

        # TODO: Learn policy based on state, action, reward
        #Q-Learning: Updating Q for previous state
        #if the old state is in Q, update its values
        if ({str(self.old_state)}.issubset(Q)) :
            Q[str(self.old_state)][str(self.old_action)] = Q[str(self.old_state)][str(self.old_action)] + self.alpha * (reward + self.gamma * (max(Q[str(self.state)].values())) - Q[str(self.old_state)][str(self.old_action)])

        #The current action will be used as old action in the next update
        self.old_action = self.action
        #The current state will be the old state in the next update
        self.old_state = self.state
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #get location and destination, compare them, and if they are equal append 1 to successes
        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]
        #deadline = self.env.agent_states[self]["deadline"]
        if location == destination:
            self.success = 1

    #set the alpha, gamma and epsilon parameters
    def setParameters(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

#modified run method, so the alpha, gamma, and epsilon can be set for the agent, so 
#testing of various parameter value combinations can be automated
def run(alpha, gamma, epsilon):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    a.setParameters(alpha, gamma, epsilon)
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0002, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    return [a.alpha, a.gamma, a.epsilon, (np.sum(a.successes[(len(a.successes)-10):len(a.successes)]) == 10), a.successes.count(1), a.totalRewards, a.rewardsPerStep, a.rewardsTable, a.moves ]

if __name__ == '__main__':
    results = []
    #run the simulation for different values of alpha gamma and epsilon
    #append the results to the results array
    for alpha in np.arange(0.0, 1.1, 0.1):
        for gamma in np.arange(0.0, 1.1, 0.1):
            for epsilon in np.arange(0.0, 0.2, 0.05):
                for i in range(100):
                    results.append(run(alpha, gamma, epsilon))
                    
    #turn the results array into a pandas DataFrame                
    df_results = pd.DataFrame(results, columns = ['alpha', 'gamma', 'epsilon', 'Last 10 Runs Successful', 'successes %','sum of rewards', 'rewards per step', 'rewards', 'number of moves'])
    #export the results to an excel file                
    df_results.to_excel('agent_results.xlsx')