import pickle
import random
from collections import defaultdict
from pathlib import Path

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]
Q_TABLE_PATH = Path(__file__).with_name("q_table.pkl")


def display_board(board):
    cells = []
    for idx, v in enumerate(board):
        cells.append(str(idx) if v == " " else v)
    print("--+---+--")
    for i in range(0, 9, 3):
        print(f"{cells[i]} | {cells[i+1]} | {cells[i+2]}")
        print("--+---+--")


def empty_cells(board):
    return [i for i, v in enumerate(board) if v == " "]


def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return all(v != " " for v in board)


class SelfPlayAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.q = defaultdict(dict)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def encode_state(self, board, player):
        opponent = "O" if player == "X" else "X"
        encoded = []
        for v in board:
            if v == player:
                encoded.append("P")
            elif v == opponent:
                encoded.append("E")
            else:
                encoded.append("_")
        return tuple(encoded)

    def choose_action(self, board, player, explore=True):
        state = self.encode_state(board, player)
        moves = empty_cells(board)
        if explore and random.random() < self.epsilon:
            return random.choice(moves), state
        values = self.q[state]
        best = max(moves, key=lambda m: values.get(m, 0.0))
        return best, state

    def update(self, state, action, reward, next_state, done):
        values = self.q[state]
        old = values.get(action, 0.0)
        if done:
            target = reward
        else:
            next_values = self.q[next_state]
            next_best = max(next_values.values()) if next_values else 0.0
            target = reward - self.gamma * next_best
        values[action] = old + self.alpha * (target - old)

    def load(self, path):
        if not path.exists():
            return False
        with path.open("rb") as fh:
            data = pickle.load(fh)
        self.q = defaultdict(dict, data)
        return True

    def save(self, path):
        with path.open("wb") as fh:
            pickle.dump(dict(self.q), fh)

    def train(self, episodes=30000):
        for episode in range(episodes):
            board = [" "] * 9
            player = "X" if episode % 2 == 0 else "O"
            while True:
                action, state = self.choose_action(board, player, explore=True)
                board[action] = player
                winner = check_winner(board)
                done = bool(winner) or is_full(board)
                reward = 1.0 if winner == player else 0.0
                if done:
                    self.update(state, action, reward, None, True)
                    break
                next_player = "O" if player == "X" else "X"
                next_state = self.encode_state(board, next_player)
                self.update(state, action, reward, next_state, False)
                player = next_player

    def best_move(self, board, player):
        action, _ = self.choose_action(board, player, explore=False)
        return action


def human_move(board, player):
    while True:
        try:
            tgt = int(input("0~8の空いている座標を入力してください: ").strip())
        except ValueError:
            print("数字で入力してください。")
            continue
        if tgt not in range(9):
            print("0~8の範囲で入力してください。")
            continue
        if board[tgt] != " ":
            print("そのマスは埋まっています。別の場所を選んでください。")
            continue
        board[tgt] = player
        return


def play_against_agent(agent, episodes):
    loaded = agent.load(Q_TABLE_PATH)
    if loaded:
        print("保存済みの学習結果を読み込みました。")
    else:
        print(f"自己対戦で{episodes}ゲーム学習中...")
        agent.train(episodes)
        agent.save(Q_TABLE_PATH)
        print("学習完了。コンピュータと対戦できます。")

    human = "O"
    cpu = "X"

    while True:
        board = [" "] * 9
        first = input("先手を打ちますか？(y/n): ").strip().lower() == "y"
        current = human if first else cpu
        display_board(board)
        while True:
            if current == human:
                print("あなたの番 (O)")
                human_move(board, human)
            else:
                move = agent.best_move(board, cpu)
                print(f"コンピュータの番 (X) -> {move}")
                board[move] = cpu

            display_board(board)
            win = check_winner(board)
            if win:
                print(f"{win} の勝ち")
                break
            if is_full(board):
                print("引き分け")
                break
            current = human if current == cpu else cpu

        retry = input("もう一度プレイしますか？(y/n): ").strip().lower()
        if retry != "y":
            break


if __name__ == "__main__":
    agent = SelfPlayAgent()
    play_against_agent(agent, episodes=30000)
