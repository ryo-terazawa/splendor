from module import GameState
import random
from mcts import SplendorNet, self_play_train, StrongSplendorNet

import torch

def random_agent(state):
    actions = state.get_legal_actions()
    return random.choice(actions)

def print_state(state):
    print("=================================")
    print(f"Current Player: {state.current_player}")
    print(f"Bank: {state.bank}")
    for lv in range(3):
        print(f"Table Level {lv+1}: {[str(t) for t in state.table[lv]]}")
    for i, p in enumerate(state.players):
        print(f"Player {i}: points={p.points}, bonuses={p.bonuses}, tokens={p.tokens}, reserved={len(p.reserved)}")
    print("=================================")


def human_agent(state, player_idx=None):
    actions = state.get_legal_actions()
    print_state(state)
    print("Available actions:")
    for idx, action in enumerate(actions):
        print(f"  {idx}: {action}")
    while True:
        try:
            choice = int(input(f"Select action (0-{len(actions)-1}): "))
            if 0 <= choice < len(actions):
                return actions[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid integer.")


def agent_action(state, model, device):
    from module import state_to_vector, action_to_id, ACTION_SIZE
    import numpy as np
    actions = state.get_legal_actions()
    state_vec = state_to_vector(state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, _ = model(state_vec)
    policy = policy.squeeze(0).cpu().numpy()
    # Mask illegal actions
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for a in actions:
        mask[action_to_id(a)] = 1.0
    policy = policy * mask
    if policy.sum() > 0:
        policy /= policy.sum()
        action_id = np.argmax(policy)
    else:
        action_id = action_to_id(actions[0])
    # Find the action corresponding to action_id
    for a in actions:
        if action_to_id(a) == action_id:
            return a
    return actions[0]

def get_player_type(idx):
    print(f"Configure Player {idx}:")
    print("  1: Human")
    print("  2: Agent (load model)")
    print("  3: Random")
    while True:
        sel = input("Select player type (1-3): ")
        if sel == "1":
            return ("human", None)
        elif sel == "2":
            model_path = input("Enter model path (.pth): ")
            return ("agent", model_path)
        elif sel == "3":
            return ("random", None)
        else:
            print("Invalid input.")

def play_multi_player():
    from mcts import StrongSplendorNet, load_model
    num_players = 0
    while num_players not in [2,3,4]:
        try:
            num_players = int(input("Number of players (2-4): "))
        except ValueError:
            pass
    player_types = []
    player_models = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(num_players):
        ptype, model_path = get_player_type(i)
        player_types.append(ptype)
        if ptype == "agent":
            model = StrongSplendorNet().to(device)
            model = load_model(model, model_path, map_location=device)
            player_models.append(model)
        else:
            player_models.append(None)
    state = GameState(num_players=num_players)
    turn = 0
    print("Player order:")
    for i, t in enumerate(player_types):
        print(f"  Player {i}: {t}")
    while not state.game_over and turn < 1000:
        turn += 1
        cur = state.current_player
        print(f"Turn {turn}, Player {cur}")
        if player_types[cur] == "human":
            action = human_agent(state, cur)
        elif player_types[cur] == "agent":
            action = agent_action(state, player_models[cur], device)
            print(f"Agent action: {action}")
        elif player_types[cur] == "random":
            action = random_agent(state)
            print(f"Random action: {action}")
        else:
            raise ValueError("Unknown player type")
        state.step(action)
    print("GAME OVER")
    for i, p in enumerate(state.players):
        print(f"Player {i}: points={p.points}")
    winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
    print(f"Winner: Player {winner}")

def play_human_vs_agent(model, device):
    state = GameState(num_players=2)
    turn = 0
    print("You are Player 0. Agent is Player 1.")
    while not state.game_over and turn < 1000:
        turn += 1
        print(f"Turn {turn}, Player {state.current_player}")
        if state.current_player == 0:
            action = human_agent(state)
        else:
            action = agent_action(state, model, device)
            print(f"Agent action: {action}")
        state.step(action)
    print("GAME OVER")
    for i, p in enumerate(state.players):
        print(f"Player {i}: points={p.points}")
    winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
    print(f"Winner: Player {winner}")

def play_game():
    state = GameState(num_players=2)

    turn = 0
    while not state.game_over and turn < 1000:  # ターン数制限を追加
        turn += 1
        action = random_agent(state)
        print(f"Turn {turn}, Player {state.current_player}, Action: {action}")
        state.step(action)
        # print_state(state)

    print("GAME OVER")
    winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
    print(f"Winner: Player {winner}")


if __name__ == "__main__":
    print("Select mode:")
    print("1: Multi-player (2-4, human/agent/random selectable)")
    print("2: Random vs Random (test)")
    print("3: Self-play training (agent vs agent)")
    mode = input("Enter mode number: ")
    if mode == "1":
        play_multi_player()
    elif mode == "2":
        play_game()
    elif mode == "3":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StrongSplendorNet().to(device)
        self_play_train(model, num_games=10, num_simulations=50, epochs=5, batch_size=32, lr=1e-3, device=device)
    else:
        print("Invalid mode.")