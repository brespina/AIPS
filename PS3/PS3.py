import numpy as np

# Transition probabilities
transition_probs = {
    's0': {'a1': {'s0': 0.8, 's1': 0.2}, 'a2': {'s0': 0.8, 's1': 0.2}},
    's1': {'a1': {'s0': 0.8, 's1': 0.2}, 'a2': {'s0': 0.8, 's1': 0.2}}
}

# Immediate rewards
rewards = {
    's0': {'a1': 1, 'a2': -1},
    's1': {'a1': -1, 'a2': 1}
}

# Discount factor
gamma = 0.9


# Function to calculate value function using Bellman Expectation Equation
def calculate_value_function(policy):
    states = list(transition_probs.keys())
    num_states = len(states)
    v_pi = np.zeros(num_states)  # Initializing value function for each state

    for i, state in enumerate(states):
        action = policy[state]
        for next_state in transition_probs[state][action]:
            prob = transition_probs[state][action][next_state]
            reward = rewards[state][action]
            v_pi[i] += prob * (reward + gamma * v_pi[states.index(next_state)])

    return v_pi


# Policy 1: pi(s) = a2 for any state s
policy1 = {'s0': 'a2', 's1': 'a2'}
value_func_policy1 = calculate_value_function(policy1)
print("Value function for policy pi(s) = a2 for any state s:", value_func_policy1)

# Policy 2: Uniformly random policy
uniform_policy = {'s0': 'a1', 's1': 'a2'}  # For demonstration, considering actions a1 and a2 equally likely
value_func_uniform_policy = calculate_value_function(uniform_policy)
print("Value function for uniformly random policy:", value_func_uniform_policy)

# Policy evaluation for gamma = 0.0
gamma_zero_policy = {'s0': 'a2', 's1': 'a2'}
gamma_zero_value_func = calculate_value_function(gamma_zero_policy)
print("Value function for gamma = 0.0:", gamma_zero_value_func)
