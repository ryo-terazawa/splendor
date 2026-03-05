from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import action_to_id, state_to_vector, ACTION_SIZE, STATE_SIZE, GameState
import os
from glob import glob

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


def mcts_with_nn(
    state,
    model,
    num_simulations=50,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    temperature=1.0,
    move_count=0,
):

    device = next(model.parameters()).device

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = {}   # action -> Node
            self.N = 0
            self.W = 0.0
            self.P = None       # prior probability
            self.expanded = False

        def Q(self):
            return self.W / (self.N + 1e-8)

    # -----------------------------
    # NN評価
    # -----------------------------
    def evaluate(node, add_noise=False):
        state_vec = state_to_vector(node.state).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = model(state_vec)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        legal_actions = node.state.get_legal_actions()
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)

        for a in legal_actions:
            mask[action_to_id(a)] = 1.0

        policy = policy * mask

        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            policy = mask / mask.sum()

        # Rootのみノイズ
        if add_noise:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            noise_masked = np.zeros(ACTION_SIZE)
            for i, a in enumerate(legal_actions):
                noise_masked[action_to_id(a)] = noise[i]
            policy = (1 - dirichlet_eps) * policy + dirichlet_eps * noise_masked

        node.P = policy
        node.expanded = True

        # 全合法手を展開する
        for action in legal_actions:
            next_state = node.state.copy()
            next_state.step(action)
            node.children[action] = Node(next_state, parent=node)

        return value

    # -----------------------------
    # Root作成
    # -----------------------------
    root = Node(state)
    evaluate(root, add_noise=True)

    # -----------------------------
    # MCTSループ
    # -----------------------------
    for _ in range(num_simulations):

        node = root
        path = [node]

        # ---- Selection ----
        while node.children:
            total_N = sum(child.N for child in node.children.values())
            best_score = -float("inf")
            best_action = None

            for action, child in node.children.items():
                a_id = action_to_id(action)

                U = (c_puct * node.P[a_id] * math.sqrt(total_N + 1) / (1 + child.N))
                Q = child.Q()
                score = Q + U

                if score > best_score:
                    best_score = score
                    best_action = action

            node = node.children[best_action]
            path.append(node)

            if not node.expanded:
                break

        # ---- Expansion + Evaluation ----
        if not node.expanded:
            v = evaluate(node)
        else:
            # terminal node
            v = 0.0

        # ---- Backprop ----
        for n in reversed(path):
            n.N += 1
            n.W += v   # 多人数対応（符号反転なし）

    # -----------------------------
    # 行動選択
    # -----------------------------
    visits = np.zeros(ACTION_SIZE, dtype=np.float32)

    for action, child in root.children.items():
        visits[action_to_id(action)] = child.N

    if visits.sum() == 0:
        legal_actions = state.get_legal_actions()
        action = legal_actions[0] if legal_actions else None
        return action, visits.tolist()

    # ---- Temperature制御 ----
    if temperature > 0 and move_count < 20:
        visits = visits ** (1.0 / temperature)
        visits /= visits.sum()
        action_id = np.random.choice(range(ACTION_SIZE), p=visits)
    else:
        action_id = np.argmax(visits)
        visits /= visits.sum()

    # id -> action
    selected_action = None
    for a in state.get_legal_actions():
        if action_to_id(a) == action_id:
            selected_action = a
            break
    
    # print("visit sum:" + str(visits.sum()) + " visit max:" + str(np.max(visits)))

    return selected_action, visits.tolist()

def self_play_train(model, num_games=200, num_simulations=100, epochs=10, batch_size=256, lr=1e-3, device="cpu"):
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
        move_count = 0
        while not state.game_over and len(history) < 1000:  # ターン数制限を追加
            # MCTSで合法手のみ選択
            action, policy = mcts_with_nn(state, model, num_simulations, move_count=move_count)
            legal_actions = state.get_legal_actions()
            # 非合法手はpass扱い
            legal_ids = [action_to_id(a) for a in legal_actions]

            filtered_policy = [
                policy[i] if i in legal_ids else 0.0
                for i in range(ACTION_SIZE)
            ]
            s = sum(filtered_policy)
            if s > 0:
                filtered_policy = [p/s for p in filtered_policy]
            else:
                filtered_policy = [
                    1.0 / len(legal_ids) if i in legal_ids else 0.0
                    for i in range(ACTION_SIZE)
                ]
            # 状態・ポリシー記録
            states.append(state_to_vector(state).clone())
            policies.append(torch.tensor(filtered_policy, dtype=torch.float32))
            history.append((state.current_player, len(states)-1))
            move_count += 1
            state.step(action)
        # 勝者決定
        winner = max(range(len(state.players)), key=lambda i: state.players[i].points)
        # 各手番に勝敗ラベル付与
        for player, idx in history:
            values.append(torch.tensor([1.0 if player == winner else -1.0], dtype=torch.float32))
        print(f"Game {game+1}/{num_games} 終了 勝者: Player {winner} ターン数: {len(history)}")

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

    for game in range(num_games):

        state = GameState(num_players=2)
        history = []
        move_count = 0

        while not state.game_over and move_count < 200:

            action, policy = mcts_with_nn(
                state,
                model,
                num_simulations=num_simulations,
                move_count=move_count,
            )

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

            state.step(action)
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
        data = torch.load(path)
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
