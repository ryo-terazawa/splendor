from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import action_to_id, state_to_vector, ACTION_SIZE, STATE_SIZE, GameState
import os
from glob import glob


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


def save_model(model, path):
    """モデルの重みを保存"""
    torch.save(model.state_dict(), path)

def load_model(model, path, map_location=None):
    """モデルの重みをロード"""
    # パスが有効なファイルかを確認
    if not os.path.isfile(path):
        raise ValueError(f"Invalid model path: {path}")
    # 可能であれば weights_only=True を利用して安全に state_dict をロード
    try:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 古い PyTorch では weights_only がサポートされていないため、従来の挙動にフォールバック
        state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

def train(model, dataset, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    SplendorNetを訓練する関数。
    dataset: (state, policy, value)のタプルを返すDataset
    """
    model = model.to(device)
    # DataLoaderのワーカ数を増やす（CPUコア数に応じて調整）
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_value = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device=="cuda" else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        for i, (state, target_policy, target_value) in enumerate(dataloader):
            state = state.to(device, non_blocking=True)
            target_policy = target_policy.to(device, non_blocking=True)
            target_value = target_value.to(device, non_blocking=True)

            optimizer.zero_grad()
            if device=="cuda":
                with torch.cuda.amp.autocast():
                    pred_policy_logits, pred_value = model(state)
                    log_probs = torch.log_softmax(pred_policy_logits, dim=1)
                    loss_policy = -torch.sum(
                        target_policy * log_probs,
                        dim=1
                    ).mean()
                    loss_value = loss_fn_value(pred_value, target_value)
                    loss = loss_policy + loss_value
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_policy_logits, pred_value = model(state)
                log_probs = torch.log_softmax(pred_policy_logits, dim=1)
                # print("\n[Debug] Batch data:")
                # print("target_policy shape:", target_policy.shape)
                # print("target_policy mean:", target_policy.mean().item())
                # print("target_policy max :", target_policy.max().item())
                # print("target_policy min :", target_policy.min().item())
                # print("log_probs shape:", log_probs.shape)
                # print("log_probs mean:", log_probs.mean().item())
                # print("log_probs max :", log_probs.max().item())
                # print("log_probs min :", log_probs.min().item())
                loss_policy = -torch.sum(
                    target_policy * log_probs,
                    dim=1
                ).mean()
                loss_value = loss_fn_value(pred_value, target_value)
                loss = loss_policy + loss_value
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            # 100バッチごとに進捗表示
            if (i+1) % 100 == 0:
                print(f"  Batch {i+1}: Loss={loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
        print(f"  Total Loss: {total_loss/len(dataloader):.4f}")
        print(f"  Policy Loss: {total_policy_loss/len(dataloader):.4f}")
        print(f"  Value  Loss: {total_value_loss/len(dataloader):.4f}")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + identity)


class StrongSplendorNet(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4):
        super().__init__()

        self.input_layer = nn.Linear(STATE_SIZE, hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # policy head
        self.policy_fc = nn.Linear(hidden_dim, ACTION_SIZE)

        # value head
        self.value_fc1 = nn.Linear(hidden_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for block in self.res_blocks:
            x = block(x)

        policy_logits = self.policy_fc(x)

        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0

class MCTS:

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def run(self, root_state, num_simulations):

        root = Node(root_state.clone())

        # NN evaluation
        state_vec = state_to_vector(root.state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(state_vec)

        policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        value = value.item()

        legal_actions = root.state.get_legal_actions()

        # Dirichlet noise
        noise = np.random.dirichlet(
            [DIRICHLET_ALPHA] * len(legal_actions)
        )

        for i, action in enumerate(legal_actions):
            a_id = action_to_id(action)
            p = policy[a_id]
            p = (1 - DIRICHLET_EPS) * p + DIRICHLET_EPS * noise[i]
            child_state = root.state.clone()
            child_state.step(action)
            root.children[action] = Node(
                child_state,
                parent=root,
                prior=p
            )

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # --- selection ---
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            # --- evaluation ---
            state = node.state

            if state.game_over:

                value = state.get_reward()

            else:

                state_vec = state_to_vector(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    policy_logits, value = self.model(state_vec)

                policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                value = value.item()

                legal_actions = state.get_legal_actions()

                for action in legal_actions:
                    a_id = action_to_id(action)
                    child_state = state.clone()
                    child_state.step(action)

                    node.children[action] = Node(
                        child_state,
                        parent=node,
                        prior=policy[a_id]
                    )

            # --- backprop ---
            self.backpropagate(search_path, value)

        return root

    def select_child(self, node):

        best_score = -1e9
        best_action = None
        best_child = None

        for action, child in node.children.items():

            u = (
                C_PUCT
                * child.prior
                * math.sqrt(node.visit_count + 1)
                / (child.visit_count + 1)
            )

            q = child.value()

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, search_path, value):

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value


def get_action_policy(root, action_size, temperature=1.0):

    visits = np.zeros(action_size)

    for action, child in root.children.items():
        a_id = action_to_id(action)
        visits[a_id] = child.visit_count

    if temperature == 0:
        best = np.argmax(visits)
        policy = np.zeros_like(visits)
        policy[best] = 1
        return policy

    visits = visits ** (1 / temperature)
    policy = visits / np.sum(visits)

    return policy

def generate_self_play_data(
    model,
    num_games=200,
    num_simulations=50,
    device="cpu",
    save_path=None,
):

    model = model.to(device)
    model.eval()

    states = []
    policies = []
    values = []


    mcts = MCTS(model, device)

    for game in range(num_games):
        state = GameState(num_players=2)
        history = []
        move_count = 0

        while not state.game_over and move_count < 200:
            # MCTSクラスを使って探索
            root = mcts.run(state, num_simulations)
            policy = get_action_policy(root, ACTION_SIZE, temperature=1.0)
            # 合法手のみ抽出
            legal_actions = state.get_legal_actions()
            legal_ids = [action_to_id(a) for a in legal_actions]
            filtered_policy = [
                policy[i] if i in legal_ids else 0.0
                for i in range(ACTION_SIZE)
            ]
            s = sum(filtered_policy)
            if s > 0:
                filtered_policy = [p / s for p in filtered_policy]
            else:
                filtered_policy = [
                    1.0 / len(legal_ids) if i in legal_ids else 0.0
                    for i in range(ACTION_SIZE)
                ]
            states.append(state_to_vector(state).clone())
            policies.append(torch.tensor(filtered_policy, dtype=torch.float32))
            history.append((state.current_player, len(states) - 1))
            # 行動選択
            # 最大値のaction_idを合法手から逆引き
            action_id = int(np.random.choice(range(ACTION_SIZE), p=filtered_policy))
            selected_action = None
            for a in legal_actions:
                if action_to_id(a) == action_id:
                    selected_action = a
                    break
            if selected_action is None:
                selected_action = legal_actions[0]
            state.step(selected_action)
            move_count += 1

        winner = max(
            range(len(state.players)),
            key=lambda i: state.players[i].points,
        )
        for player, idx in history:
            values.append(
                torch.tensor(
                    [1.0 if player == winner else -1.0],
                    dtype=torch.float32,
                )
            )
        print(f"Game {game+1}/{num_games} 終了 勝者: Player {winner}")

    data = {
        "states": torch.stack(states),
        "policies": torch.stack(policies),
        "values": torch.stack(values),
    }

    if save_path is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"selfplay_data_{now}.pt"

    torch.save(data, save_path)

    print(f"\nデータ保存完了: {save_path}")
    print(f"総サンプル数: {len(states)}")

    return save_path

def train_from_files(
    model,
    data_paths=None,
    data_dir=None,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    device="cpu",
    max_samples=None,   # 古いデータ切り捨て用
):
    """
    複数の自己対戦データをまとめて学習する。
    data_paths: 明示的なファイルリスト
    data_dir: ディレクトリ指定（*.ptを全部読む）
    max_samples: 最新Nサンプルのみ使う（replay buffer化）
    """

    if data_dir is not None:
        data_paths = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not data_paths:
        raise ValueError("データファイルが見つかりません")

    print("ロード対象ファイル:")
    for p in data_paths:
        print("  ", p)

    all_states = []
    all_policies = []
    all_values = []

    for path in data_paths:
        data = torch.load(path, weight_only=True)
        all_states.append(data["states"])
        all_policies.append(data["policies"])
        all_values.append(data["values"])

    states = torch.cat(all_states, dim=0)
    policies = torch.cat(all_policies, dim=0)
    values = torch.cat(all_values, dim=0)

    print(f"\n総サンプル数: {len(states)}")

    # Replay Buffer制御（古いデータ削減）
    if max_samples is not None and len(states) > max_samples:
        print(f"最新 {max_samples} サンプルのみ使用")
        states = states[-max_samples:]
        policies = policies[-max_samples:]
        values = values[-max_samples:]

    class SelfPlayDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(states)

        def __getitem__(self, idx):
            return states[idx], policies[idx], values[idx]

    dataset = SelfPlayDataset()

    train(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(model, f"trained_model_{now}.pth")

    print("学習完了")
