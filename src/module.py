import random
import copy
from dataclasses import dataclass, field
import csv
from itertools import combinations

COLORS = ["white", "blue", "green", "red", "black"]
GOLD = "gold"
SETTING_MAP = {
    2:{
        'BANK_MAX':4,
        'NOBILITY_NUM':2
    },
    3:{
        'BANK_MAX':5,
        'NOBILITY_NUM':4
    },
    4:{
        'BANK_MAX':7,
        'NOBILITY_NUM':5
    },
}
COLOR_MAP = {
    "w": "white",
    "b": "blue",
    "g": "green",
    "r": "red",
    "k": "black"
}

# =========================
# Card
# =========================
@dataclass
class Card:
    level: int
    cost: dict
    bonus: str
    points: int
    # printに対応する
    def __str__(self):
        return f"Card(Lv{self.level}, {self.bonus}, {self.points}pt, cost={self.cost})"

# =========================
# Nobility
# =========================
@dataclass
class Nobility:
    cost: dict
    points: int


# =========================
# Player
# =========================
@dataclass
class Player:
    tokens: dict = field(default_factory=lambda: {c: 0 for c in COLORS + [GOLD]})
    bonuses: dict = field(default_factory=lambda: {c: 0 for c in COLORS})
    points: int = 0
    reserved: list = field(default_factory=list)

    def total_tokens(self):
        return sum(self.tokens.values())


# =========================
# GameState
# =========================
class GameState:

    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [Player() for _ in range(num_players)]
        self.current_player = 0

        # トークン供給
        self.bank = {c: SETTING_MAP[num_players]['BANK_MAX'] for c in COLORS}
        self.bank[GOLD] = 5

        # 簡易デッキ（本来はCSVなどからロード）
        self.decks = self.create_all_decks_from_csv('../data/Card.csv')
        random.shuffle(self.decks[1])
        random.shuffle(self.decks[2])
        random.shuffle(self.decks[3])

        self.table = [[] for _ in range(3)]
        for lv in range(3):
            self.table[lv] = [self.decks[lv+1].pop() for _ in range(4)]

        nobility_lists = self.create_nobility_from_csv('../data/Nobility.csv')
        random.shuffle(nobility_lists)
        self.nobility = nobility_lists[:SETTING_MAP[num_players]['NOBILITY_NUM']]
        

        self.game_over = False

    # =========================
    # デッキ生成
    # =========================
    def create_all_decks_from_csv(self, filepath):
        decks = {1: [], 2: [], 3: []}

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                level = int(row["level"])
                bonus = COLOR_MAP[row["color"]]
                points = int(row["point"])

                cost = {
                    "white": int(row["white"]),
                    "blue": int(row["blue"]),
                    "green": int(row["green"]),
                    "red": int(row["red"]),
                    "black": int(row["black"]),
                }

                decks[level].append(Card(level, cost, bonus, points))

        return decks
    
    # =========================
    # 貴族生成
    # =========================
    def create_nobility_from_csv(self, filepath):
        nobilitys = []

        with open(filepath, newline="", encoding="utf-8") as f:
            
            reader = csv.DictReader(f)
            for row in reader:
                cost = {
                    "white": int(row["white"]),
                    "blue": int(row["blue"]),
                    "green": int(row["green"]),
                    "red": int(row["red"]),
                    "black": int(row["black"]),
                }
                points = 3
                nobilitys.append(Nobility(cost,points))
            

        return nobilitys

    # =========================
    # 合法手生成
    # =========================
    def get_legal_actions(self):
        actions = []
        player = self.players[self.current_player]

        # ① トークン3枚取得（異色）
        available_colors = [c for c in COLORS if self.bank[c] > 0]
        if len(available_colors) >= 3:
            for comb in combinations(available_colors, 3):
                actions.append(("take3", comb))

        # ② 同色2枚（4枚以上ある場合）
        for c in COLORS:
            if self.bank[c] >= 4:
                actions.append(("take2", c))

        # ③ 購入
        for lv in range(3): 
            for i, card in enumerate(self.table[lv]):
                if self.can_buy(player, card):
                    actions.append((f"buy_table{lv+1}", i))

        # ④ 予約（最大3枚）
        for lv in range(3):
            if len(player.reserved) < 3:
                for i in range(len(self.table[lv])):
                    actions.append((f"reserve{lv+1}", i))

        # ⑤ 予約カード購入
        for i, card in enumerate(player.reserved):
            if self.can_buy(player, card):
                actions.append((f"buy_reserved", i))
        
        # パス
        actions.append(("pass", None))

        return actions

    # =========================
    # 購入可能判定
    # =========================
    def can_buy(self, player, card):
        for c in COLORS:
            required = card.cost[c] - player.bonuses[c]
            if required > player.tokens[c] + player.tokens[GOLD]:
                return False
        return True

    # =========================
    # 行動実行
    # =========================
    def step(self, action):
        player = self.players[self.current_player]
        action_type, value = action

        if action_type == "take3":
            for c in value:
                self.bank[c] -= 1
                player.tokens[c] += 1

        elif action_type == "take2":
            c = value
            self.bank[c] -= 2
            player.tokens[c] += 2

        elif action_type.startswith("buy_table"):
            lv = int(action_type[-1]) - 1
            card = self.table[lv].pop(value)
            self.pay_cost(player, card)
            player.bonuses[card.bonus] += 1
            player.points += card.points

            if self.decks:
                self.table[lv].append(self.decks[lv+1].pop())
        
        elif action_type == "buy_reserved":
            card = player.reserved.pop(value)
            self.pay_cost(player, card)
            player.bonuses[card.bonus] += 1
            player.points += card.points

        elif action_type.startswith("reserve"):
            lv = int(action_type[-1]) - 1
            card = self.table[lv].pop(value)
            player.reserved.append(card)
            if self.bank[GOLD] > 0:
                self.bank[GOLD] -= 1
                player.tokens[GOLD] += 1

            if self.decks:
                self.table[lv].append(self.decks[lv+1].pop())
        
        elif action_type == "pass":
            pass
        # 貴族判定
        for nob in self.nobility[:]:
            if all(player.bonuses[c] >= nob.cost[c] for c in COLORS):
                player.points += nob.points
                self.nobility.remove(nob)

        # 勝利判定（簡略：15点）
        if player.points >= 15:
            self.game_over = True

        self.current_player = (self.current_player + 1) % self.num_players

    # =========================
    # コスト支払い
    # =========================
    def pay_cost(self, player, card):
        for c in COLORS:
            required = max(0, card.cost[c] - player.bonuses[c])
            use_color = min(required, player.tokens[c])
            player.tokens[c] -= use_color
            self.bank[c] += use_color
            required -= use_color

            if required > 0 and player.tokens[GOLD] >= required:
                player.tokens[GOLD] -= required
                self.bank[GOLD] += required