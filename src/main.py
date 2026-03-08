from module import GameState
import random
from mcts import MCTS, get_action_policy
from mcts import StrongSplendorNet, generate_self_play_data, train_from_files, load_model
from module import state_to_vector, action_to_id, ACTION_SIZE
import numpy as np
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
    actions = state.get_legal_actions()
    state_vec = state_to_vector(state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(state_vec)
        policy = torch.softmax(policy_logits, dim=-1)        
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

# =========================
# MCTS Agent
# =========================
def mcts_agent_action(state, model, device, num_simulations=100):

    mcts = MCTS(model, device)

    root = mcts.run(state, num_simulations)

    policy = get_action_policy(root, ACTION_SIZE, temperature=0)

    action_id = np.argmax(policy)

    actions = state.get_legal_actions()

    for a in actions:
        if action_to_id(a) == action_id:
            return a

    return random.choice(actions)

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
            action = mcts_agent_action(state, player_models[cur], device)
            print(f"MCTS action: {action}")
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
            action = mcts_agent_action(state, model, device)
            print(f"MCTS action: {action}")
        state.step(action)
    print("GAME OVER")
    for i, p in enumerate(state.players):
        print(f"Player {i}: points={p.points}")
    winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
    print(f"Winner: Player {winner}")
# =========================
# Random test game
# =========================
def random_test_game():
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

# =========================
# main
# =========================
if __name__ == "__main__":
    print("Select mode:")
    print("1: Multi-player (2-4, human/agent/random selectable)")
    print("2: Random vs Random (test)")
    print("3: Self-play training (agent vs agent)")
    print("4: make train data")
    print("5: train model")
    mode = input("Enter mode number: ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StrongSplendorNet().to(device)
    if mode == "1":
        play_multi_player()
    elif mode == "2":
        random_test_game()
    elif mode == "3":
        generate_self_play_data(model, num_games=100, num_simulations=50, device=device)
    elif mode == "4":
        train_from_files(model, data_dir="data/train", device=device)
    else:
        print("Invalid mode.")