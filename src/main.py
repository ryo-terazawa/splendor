from module import GameState
import random

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
    play_game()