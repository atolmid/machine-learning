import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import collections
import pandas as pd
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    global Q

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        #create 'actions' list
        self.actions = [None, 'forward', 'left', 'right']
        #initialize variables that will be used
        #gamma should probably be affected by the deadline
        self.gamma = None
        self.alpha = None
        self.epsilon = None
        self.Q = {}
        self.old_state = None
        self.old_action = None
        #self.runs = 1
        #list to use in measuring the success rate
        self.successes = []


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.runs = 1
        self.old_state = None
        self.old_action = None
        # TODO: Prepare for a new trip; reset any variables here, if required
        
    def setParameters(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        #set the state using the inputs 
        self.state = inputs
        #add the waypoint to the state
        self.state['waypoint'] = str(self.next_waypoint)
        #when apha is chosen to decay, it will be calculated as follows
        #alpha = 1.0/(self.runs + 1)
        
        # TODO: Select action according to your policy
        #if the current state is not in our Q table, set random Q values for all
        #current state/action pairs
        if str(self.state) not in self.Q.keys():
            dict1 = {}
            for act in self.actions:
                dict1[str(act)] =  random.random() - 0.5
            self.Q[str(self.state)] = dict1
            
        #random actions (for the initial part of the assignment with no Q-Learning)
        dict1 = {}
        for act in self.actions:
            dict1[str(act)] =  random.random() - 0.5
        self.Q[str(self.state)] = dict1

        #using an exploration rate (epsilon) for the choice of next action
        #epsilon will be decreasing at every run
        #epsilon = 1.0/(2+self.runs)

        #choose a random number, and if it is less than epsilon, choose a random action
        #otherwise, choose action with the highest Q value
        if random.random()<self.epsilon:
            action = self.actions[int(4*random.random())]
        else:
            action = max(self.Q[str(self.state)], key=lambda v: self.Q[str(self.state)][v])
        #if the chosen action is None, we have to change it, since it cannot be a string
        if action == 'None':
            action = None

        #increase the runs counter (is used for parameter - alpha, etc - decaying) 
        self.runs += 1
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        #if there is no old_state data (first run), use random Q values
        if str(self.old_state) not in self.Q.keys():
            dict1 = {}
            for act in self.actions:
                dict1[str(act)] =  random.random() - 0.5
            self.Q[str(self.old_state)] = dict1
                 
        #Q-Learning: Updating Q for previous state (using epsilon)
        #q-new = (1- alpha)*q-old + alpha*[reward + gamma * q-max]
        self.Q[str(self.old_state)][str(self.old_action)] = self.Q[str(self.old_state)][str(self.old_action)] + self.alpha * (reward + self.gamma * (max(self.Q[str(self.state)].values())) -self.Q[str(self.old_state)][str(self.old_action)])
        #The current action will be used as old action in the next update
        self.old_action = action
        #The current state will be the old state in the next update
        self.old_state = self.state
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        location = self.env.agent_states[self]["location"]
        #print "location : ", location
        destination = self.env.agent_states[self]["destination"]
        #print "destination : ", destination
        if location == destination:
            self.successes.append(1)
        else:
            self.successes.append(0)


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
    return [a.alpha, a.gamma, a.epsilon, (100.0*a.successes.count(1)/(a.successes.count(1)+a.successes.count(0))), a.successes.count(1),  a.successes.count(0), a.successes]#, "\n"]
    #print "number of runs :", len(a.successes) 
    #print "number of successes :", a.successes.count(1) 

if __name__ == '__main__':
    results = []
    #run the simulation for different values of alpha gamma and epsilon
    #append the results to the results array
    for alpha in np.arange(0.0, 1.1, 0.1):
        for gamma in np.arange(0.0, 1.1, 0.1):
            for epsilon in np.arange(0.0, 0.2, 0.05):
                for i in range(100):
                    #sim_results = run() 
                    results.append(run(alpha, gamma, epsilon))
    #for i in range(100): 
        #results.append(run(0.6, 0.6, 0.1))
                    
    #turn the results array into a pandas DataFrame                
    df_results = pd.DataFrame(results, columns = ['alpha', 'gamma', 'epsilon', 'success %', 'success', 'failure', 'success table'])
    print df_results
    #print the row that has the highest success rate for the smart cab
    print "row with the highest success rate : ", df_results.ix[df_results['success %'].idxmax()]
    #export the results to an excel file                
    df_results.to_excel('original_agent_results.xlsx')
