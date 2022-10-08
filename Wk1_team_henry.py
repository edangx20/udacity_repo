import random
from typing import Dict, List, Tuple

from tqdm import tqdm

from game_mechanics import Cell, WildTictactoeEnv, load_dictionary, render, save_dictionary, Player, \
    choose_move_randomly, reward_function, play_wild_ttt_game, human_player
from check_submission import check_submission

TEAM_NAME = "Henry"  # <---- Enter your team name here!


def train() -> Dict:
    """
    Returns:
        A dictionary containing data to be used by your agent during gameplay.
        You can structure this however you like as long as you write a
        choose_move function that can use it.
    """
    # Set hyperparameters
    alpha = 0.25
    alpha_decay = 0.999999
    epsilon = 0.2
    epsilon_decay = 0.999999

    # This is the default value of each state (the value at initialisation)
    default = 0

    # Initialise empty value function
    value_fn = {}

    def epsilon_greedy_opponent(state: List[str]):
        """Epsilon greedy policy for opponent, using a larger epsilon"""
        return choose_move_randomly(state) if random.random() < epsilon * 3 + 0.4 else choose_move(state, value_fn)

    game = WildTictactoeEnv(epsilon_greedy_opponent)

    # Train for 1,000,000 episodes
    for _ in tqdm(range(1000000)):

        state, reward, done, info = game.reset()

        while not done:
            prev_state = str(state)
            # Use epsilon-greedy to take moves (rewritten so we use a smaller epsilon than the opponent)
            action = choose_move_randomly(state) if random.random() < epsilon else choose_move(state, value_fn)

            state, reward, done, info = game.step(action)

            if reward:
                # The alpha * (r_t + v(s_t+1)) term is negative because players alternate
                value_fn[prev_state] = (1 - alpha) * value_fn.get(prev_state, default) - alpha * reward
            else:
                value_fn[prev_state] = (1 - alpha) * value_fn.get(prev_state, default) + alpha * value_fn.get(str(state), default)

        # We choose to decay alpha so we take smaller update steps as we converge
        #  to the optimal policy & value function
        alpha *= alpha_decay

        # Decaying epsilon so we play closer to greedily over time,
        #  since the policy followed affects the value function
        epsilon *= epsilon_decay

    validate_value_fn(value_fn)

    return value_fn


def choose_move(board: List, value_function: Dict) -> Tuple[int, str]:
    """This function will be called during competitive play. It will take the current state of the
    board. You will need to return a single move to play on this board.

    Args:
        board: list representing the board.

            Elements which are 0 are empty spaces.
            Elements which are 1 are YOUR crosses.
            Elements which are -1 are THE OPPONENT's noughts.

        Example input: [0, 0, 1, -1, 0, 1, -1, 0, 0]
            Where above represents:

                |   | X
             -----------
              O |   | X
             -----------
              O |   |

        value_function: The dictionary you saved in your training function.

    Returns:
        position (int): The position you want to place your piece in (an integer 0 -> 8),
                        where 0 is the top left square and 8 is the bottom right.
        counter (Cell): The counter you want to use to place your piece. Either Cell.X or Cell.O.
    """
    poss_positions: List[int] = [count for count, item in enumerate(board) if item == Cell.EMPTY]
    counters: List[str] = [Cell.O, Cell.X]

    # Below picks randomly between highest value successor states
    max_value = -1000
    best_moves = []

    for pos in poss_positions:
        for counter in counters:
            # .copy() stops the changes we make from affecting the actual board
            state = board.copy()
            state[pos] = counter

            action_value = value_function.get(str(state), 0) + reward_function(state)

            if action_value > max_value:
                max_value = action_value
                best_moves = [(pos, counter)]
            elif action_value == max_value:
                best_moves.append((pos, counter))

    return random.choice(best_moves)


def validate_value_fn(my_dict: Dict):
    """Test your algorithm here!

    Args:
        my_dict (Dict): the dictionary you returned from your training function.

    The example below plays a single game of wild tictactoe against itself, think about
    how you might want to adapt this to test the performance of your algorithm.
    """
    for oppo_name, oppo_policy in [("Random", choose_move_randomly), ("Yourself", lambda x: choose_move(x, my_dict))]:
        game = WildTictactoeEnv(oppo_policy)
        num_won, num_drawn, num_lost = 0, 0, 0
        for _ in range(1000):
            observation, reward, done, info = game.reset()
            while not done:
                action = choose_move(observation, my_dict)
                observation, reward, done, info = game.step(action)

            if reward == 0:
                num_drawn += 1
            elif reward == 1:
                num_won += 1
            else:
                num_lost += 1

        print(f"Vs {oppo_name}.\nWon: {num_won}, drawn: {num_drawn}, lost: {num_lost}")


if __name__ == "__main__":

    value_fn = train()
    save_dictionary(value_fn, TEAM_NAME)
    # value_fn = load_dictionary(TEAM_NAME)
    # validate_value_fn(value_fn)
    # render(choose_move, value_fn)
    # check_submission(TEAM_NAME)
    value_fn = load_dictionary(TEAM_NAME)

    play_wild_ttt_game(
        your_choose_move=human_player,
        opponent_choose_move=lambda x: choose_move(x, value_fn),
        render=True,
    )