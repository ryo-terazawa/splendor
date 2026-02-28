from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import action_to_id, state_to_vector, ACTION_SIZE, STATE_SIZE, GameState


class DummyDataset(torch.utils.data.Dataset):
    """
    ダミーデータセット。実際のデータで置き換えてください。
    各サンプルは(state_vector, (policy, value))のタプル。
    """
    def __init__(self, num_samples=100):
        self.states = torch.randn(num_samples, STATE_SIZE)
        self.policies = torch.softmax(torch.randn(num_samples, ACTION_SIZE), dim=1)
        self.values = torch.tanh(torch.randn(num_samples, 1))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def save_model(model, path):
    """モデルの重みを保存"""
    torch.save(model.state_dict(), path)

def load_model(model, path, map_location=None):
    """モデルの重みをロード"""
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

def train(model, dataset, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    SplendorNetを訓練する関数。
    dataset: (state, policy, value)のタプルを返すDataset
    """
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_policy = nn.KLDivLoss(reduction="batchmean")
    loss_fn_value = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for state, target_policy, target_value in dataloader:
            state = state.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = model(state)
            loss_policy = loss_fn_policy(pred_policy.log(), target_policy)
            loss_value = loss_fn_value(pred_value, target_value)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")


class SplendorNet(nn.Module):
    def __init__(self):
        super(SplendorNet, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, ACTION_SIZE)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value
    
    
def mcts_with_nn(state, model, num_simulations=100):
    # MCTSとNN推論の連携
    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = {}
            self.N = 0  # 訪問回数
            self.W = 0  # 累積価値
            self.P = None  # NNポリシー
            self.V = None  # NNバリュー

    root = Node(state)
    # ルートでNN推論
    state_vec = torch.tensor(state_to_vector(state), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = model(state_vec)
        policy = policy_logits.squeeze(0).cpu().numpy()
        value = value.item()
    legal_actions = state.get_legal_actions()
    mask = [1 if a in legal_actions else 0 for a in range(ACTION_SIZE)]
    policy = policy * mask
    if policy.sum() > 0:
        policy = policy / policy.sum()
    else:
        policy = [1/len(legal_actions) if a in legal_actions else 0 for a in range(ACTION_SIZE)]
    root.P = policy
    root.V = value

    c_puct = 1.0

    for _ in range(num_simulations):
        node = root
        path = [node]
        # 選択
        while node.children:
            total_N = sum(child.N for child in node.children.values())
            best_score = -float('inf')
            best_a = None
            for a, child in node.children.items():
                a_id = action_to_id(a)
                u = c_puct * node.P[a_id] * math.sqrt(total_N+1) / (1 + child.N)
                q = child.W / (child.N+1e-8)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_a = a
            node = node.children[best_a]
            path.append(node)
        # 展開
        if not node.state.game_over:
            legal_actions = node.state.get_legal_actions()
            for a in legal_actions:
                if a not in node.children:
                    next_state = node.state.copy()
                    next_state.step(a)
                    child = Node(next_state, parent=node)
                    # NN推論
                    state_vec = torch.tensor(state_to_vector(next_state), dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        p_logits, v = model(state_vec)
                        p = p_logits.squeeze(0).cpu().numpy()
                        v = v.item()
                    mask = [1 if aa in next_state.get_legal_actions() else 0 for aa in range(ACTION_SIZE)]
                    p = p * mask
                    if p.sum() > 0:
                        p = p / p.sum()
                    else:
                        p = [1/len(mask) if m else 0 for m in mask]
                    child.P = p
                    child.V = v
                    node.children[a] = child
            # 1つだけ子ノードを選んで評価値をバックプロパゲーション
            if node.children:
                a = legal_actions[0]
                v = node.children[a].V
            else:
                v = 0
        else:
            v = 0
        # バックプロパゲーション
        for n in reversed(path):
            n.N += 1
            n.W += v

    # 最も訪問回数の多い手を選択
    visits = [root.children[a].N if a in root.children else 0 for a in range(ACTION_SIZE)]
    if sum(visits) > 0:
        best_action = int(np.argmax(visits))
        policy = np.array(visits, dtype=np.float32)
        policy = policy / policy.sum()
    else:
        legal_actions = state.get_legal_actions()
        best_action = legal_actions[0] if legal_actions else None
        policy = np.array([1/len(legal_actions) if a in legal_actions else 0 for a in range(ACTION_SIZE)], dtype=np.float32)
    return best_action, policy.tolist()

def self_play_train(model, num_games=10, num_simulations=50, epochs=5, batch_size=32, lr=1e-3, device="cpu"):
    """
    MCTS同士で自己対戦し、合法手のみでデータを収集し学習する関数。
    非合法手はpass扱い。
    """
    model = model.to(device)
    states = []
    policies = []
    values = []

    for game in range(num_games):
        state = GameState(num_players=2)
        history = []
        while not state.game_over and len(history) < 1000:  # ターン数制限を追加
            # MCTSで合法手のみ選択
            action, policy = mcts_with_nn(state, model, num_simulations)
            legal_actions = state.get_legal_actions()
            # 非合法手はpass扱い
            filtered_policy = [p if a in legal_actions else 0 for a, p in enumerate(policy)]
            s = sum(filtered_policy)
            if s > 0:
                filtered_policy = [p/s for p in filtered_policy]
            else:
                filtered_policy = [1/len(legal_actions) if a in legal_actions else 0 for a in range(ACTION_SIZE)]
            # 状態・ポリシー記録
            states.append(torch.tensor(state_to_vector(state), dtype=torch.float32))
            policies.append(torch.tensor(filtered_policy, dtype=torch.float32))
            history.append((state.current_player, len(states)-1))
            state.step(action)
        # 勝者決定
        winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
        # 各手番に勝敗ラベル付与
        for player, idx in history:
            values.append(torch.tensor([1.0 if player == winner else -1.0], dtype=torch.float32))
        print(f"Game {game+1}/{num_games} 終了 勝者: Player {winner}")

    # データセット作成
    class SelfPlayDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(states)
        def __getitem__(self, idx):
            return states[idx], policies[idx], values[idx]

    dataset = SelfPlayDataset()
    train(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(model, f"self_play_model_{now}.pth")
    