from agents.agent import Agent
from gym_env import PokerEnv
import random
import numpy as np
from treys import Evaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.wins = 0
        self.total = 0
        self.evaluator = Evaluator()
        self.in_position = True

    def act(self, observation, reward, terminated, truncated, info):
        if self.total >= 1.5 * (1000 - info["hand_number"]) + 1:
            action_type = action_types.FOLD.value
            raise_amount = 0
            card_to_discard = -1
            return action_type, raise_amount, card_to_discard
        #(self.wins/(info["hand_number"]+1))
        # Log new street starts with important info
        
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
            if(observation["opp_last_action"]) == 'None':
                self.in_position = False
            else: 
                self.in_position = True
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")

        my_cards = [int(card) for card in observation["my_cards"]]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
        opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

        # Calculate equity through Monte Carlo simulation
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        non_shown_cards = [i for i in range(27) if i not in shown_cards]

        def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation
        num_simulations = 8000
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations

        # Calculate pot odds
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        

        
        self.logger.debug(f"Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")

        # Decision making
        raise_amount = 0
        card_to_discard = -1

        og_equity = equity
        # Adjusted Equity
        #print(equity)
        equity = equity * (1+(np.exp((0.5 - self.wins/(info["hand_number"]+1)))-1)*(min(info["hand_number"]+1, 100)/100))
        #print(equity)

        if observation["street"] == 0:
            equity = equity - (1.0 - equity) * ((observation["opp_bet"] * 2) / 200)
        else:
            equity = equity - (1.0 - equity) * ((observation["opp_bet"] * 2) / 100) 
        #print(equity)

        if self.in_position:
            equity *= 1.05
        else:
            equity *= 0.95
        
        #if observation["street"] == 0:
        #    print(f"our equity: {equity}  our win rate: {self.wins/(info["hand_number"]+1)}")

        # Only log very significant decisions at INFO level
        if (equity > 0.7 or equity > 0.52 and observation["street"] == 0) and observation["valid_actions"][action_types.RAISE.value] and (equity < 0.9 or random.random() < 0.65):
            raise_amount = min(int(pot_size * random.uniform(equity/2, equity*1.5)), observation["max_raise"])
            raise_amount = max(raise_amount, observation["min_raise"])
            action_type = action_types.RAISE.value
            if raise_amount > 20:  # Only log large raises
                self.logger.info(f"Large raise to {raise_amount} with equity {equity:.2f}")
        elif equity >= pot_odds and observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        elif observation["valid_actions"][action_types.CHECK.value]:
            if self.in_position:
                if random.random() < 0.65 or observation["valid_actions"][action_types.RAISE.value] == False:
                    action_type = action_types.CHECK.value
                else:
                    raise_amount = min(int(pot_size * random.uniform(0.5, 1.5)), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action_type = action_types.RAISE.value
            else: 
                if random.random() < 0.85 or observation["valid_actions"][action_types.RAISE.value] == False:
                    action_type = action_types.CHECK.value
                else:
                    raise_amount = min(int(pot_size * random.uniform(0.5, 1.5)), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action_type = action_types.RAISE.value

        elif observation["valid_actions"][action_types.DISCARD.value]:

            num_simulations = 1600 #change this for 5x

            wins = sum(
            evaluate_hand(([my_cards[0], drawn_cards[0]], opp_drawn_card + drawn_cards[1: 3 - len(opp_drawn_card)], community_cards + drawn_cards[3 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
            )
            equity1 = wins / num_simulations - 0.05

            wins = sum(
            evaluate_hand(([drawn_cards[0], my_cards[1]], opp_drawn_card + drawn_cards[1: 3 - len(opp_drawn_card)], community_cards + drawn_cards[3 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
            )
            equity2 = wins / num_simulations - 0.05

            if equity1 > og_equity and equity1 >= equity2:
                action_type = action_types.DISCARD.value
                card_to_discard = 1
                self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")
            elif equity2 > og_equity and equity1 <= equity2:
                action_type = action_types.DISCARD.value
                card_to_discard = 0
                self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")

            else:
                action_type = action_types.FOLD.value

        else:
            action_type = action_types.FOLD.value
            if observation["opp_bet"] > 20:  # Only log significant folds
                self.logger.info(f"Folding to large bet of {observation['opp_bet']}")
        
        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        #print(reward)
        self.total += reward
        if reward > 0:
            self.wins += 1
        #print(observation)
        if terminated and abs(reward) > 20:  # Only log significant hand results
            self.logger.info(f"Significant hand completed with reward: {reward}")
