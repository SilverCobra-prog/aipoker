from agents.agent import Agent
from gym_env import PokerEnv
import random
import math
import time

import treys

action_types = PokerEnv.ActionType

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Includes observation and any extra state info
        self.parent = parent
        self.action = action  # The action that led to this state from parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_actions = self.get_valid_actions()

    def get_valid_actions(self):
        # Assuming state contains observation with 'valid_actions'
        valid = self.state["observation"]["valid_actions"]
        return [i for i, is_valid in enumerate(valid) if is_valid]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.simulate_action(self.state, action)
        child = Node(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def simulate_action(self, state, action):
        """
        Simulates the result of taking an action in the current state.
        This method updates the state based on the action taken.
        """
        # Clone the current state to avoid modifying the original
        next_state = {
            "observation": state["observation"].copy(),
            "reward": state["reward"],
            "terminated": state["terminated"],
            "truncated": state["truncated"],
            "info": state["info"].copy()
        }

        # Ensure the 'pot' key exists in the info dictionary
        next_state["info"]["pot"] = next_state["info"].get("pot", 0)

        # Handle the action and update the state
        if action == action_types.RAISE.value:
            # Deduct chips for raising and add to the pot
            raise_amount = next_state["observation"].get("min_raise", 0)
            next_state["observation"]["chips"] = next_state["observation"].get("chips", 0) - raise_amount
            next_state["info"]["pot"] += raise_amount
        elif action == action_types.CALL.value:
            # Match the opponent's bet and add to the pot
            call_amount = next_state["info"].get("current_bet", 0)
            next_state["observation"]["chips"] = next_state["observation"].get("chips", 0) - call_amount
            next_state["info"]["pot"] += call_amount
        elif action == action_types.FOLD.value:
            # Folding ends the game for the player
            next_state["terminated"] = True
            next_state["reward"] = -1 
        elif action == action_types.DISCARD.value:
            # Discard a card (example logic)
            if "hand" in next_state["observation"]:
                next_state["observation"]["hand"] = next_state["observation"]["hand"][1:]  # Discard the first card

        # Check for terminal conditions (e.g., no chips left)
        if next_state["observation"].get("chips", 0) <= 0:
            next_state["terminated"] = True
            next_state["reward"] = -10

        return next_state

    def is_terminal(self):
        return self.state["terminated"]

    def best_uct_child(self, c_param=1.4):
        def uct(node):
            if node.visits == 0:
                return float("inf")
            return (node.total_reward / node.visits) + c_param * math.sqrt(math.log(self.visits) / node.visits)

        return max(self.children, key=uct)

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0

    def act(self, observation, reward, terminated, truncated, info):
        root_state = {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }

        root = Node(root_state)
        best_node = self.monte_carlo_tree_search(root, time_limit=0.05)  # 50ms budget
        action_type = best_node.action

        # Setup default values
        raise_amount = 0
        card_to_discard = -1  # -1 means no discard
        
        # If we chose to raise, pick a random amount between min and max
        if action_type == action_types.RAISE.value:
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
        if action_type == action_types.DISCARD.value:
            card_to_discard = random.randint(0, 1)
        
        return action_type, raise_amount, card_to_discard

    def monte_carlo_tree_search(self, root, time_limit=0.5):
        start_time = time.time()
        while time.time() - start_time < time_limit:
            leaf = self.traverse(root)
            result = self.rollout(leaf)
            self.backpropagate(leaf, result)
        return self.best_child(root)

    def traverse(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_uct_child()
        if not node.is_terminal() and not node.is_fully_expanded():
            return node.expand()
        return node

    def rollout(self, node):
        # Simple random rollout policy
        current = node
        depth = 0
        while not current.is_terminal() and depth < 5:
            valid = current.get_valid_actions()
            if not valid:
                break
            action = random.choice(valid)
            next_state = current.simulate_action(current.state, action)
            current = Node(next_state)
            depth += 1
        return self.evaluate(current.state)

    def evaluate(self, state):
        observation = state.get("observation", {})
        info = state.get("info", {})
        
        # Use final reward if terminal
        if state.get("terminated", False):
            return state.get("reward", 0)
        
        # Example heuristics:
        player_chips = observation.get("chips", 0)
        opp_chips = info.get("opp_chips", 0)
        pot = info.get("pot", 0)
        hand_strength = info.get("hand_strength", 0.5)  # Range 0-1 if available

        # Reward = weighted sum of hand strength and chip advantage
        score = (
            hand_strength * 100 + 
            (player_chips - opp_chips) * 0.1 + 
            pot * 0.05
        )
        return score


    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.total_reward += result
            node = node.parent

    def best_child(self, node):
        return max(node.children, key=lambda c: c.visits) if node.children else node

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")