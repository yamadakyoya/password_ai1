"""CSVを読み込み、AIを学習させ、モデル・表・グラフを保存する。"""

# 【プログラムの流れ】
# AI作成準備（設定・CSV読み込み・データ分け）
#     ↓
# AI作成（学習・モデル保存）
#     ↓
# 結果作成（表・グラフ）
#
# main()から、この順番で実行する。

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ==================================================
# 1. AI作成準備
# CSVの場所・保存先・学習件数を決める
# ==================================================

BASE_FOLDER = Path(__file__).resolve().parent

# 読み込むCSVファイル
DATA_FILE = BASE_FOLDER / "data" / "password_dataset.csv"

# 学習済みAIとグラフを保存する場所
MODEL_FOLDER = BASE_FOLDER / "model"
MODEL_FILE = MODEL_FOLDER / "password_strength_model.joblib"
GRAPH_FILE = MODEL_FOLDER / "accuracy_by_rank.png"

# 各ランクから学習用と問題用に分ける件数
TRAIN_COUNT_PER_RANK = 1800
TEST_COUNT_PER_RANK = 200

# A・B・Cの名前
RANKS = ["A", "B", "C"]

# ランダムフォレストで使う木の数
TREE_COUNT = 100

# AIに読み込ませるCSV内の数字の列
# length: パスワードの文字数
# has_sequence: 連番があれば1、なければ0
# has_repeated_pair: 同じ文字が続けば1、なければ0
# has_weak_pattern: 弱いパターンがあれば1、なければ0
AI_INPUT_COLUMNS = [
    "length",
    "has_sequence",
    "has_repeated_pair",
    "has_weak_pattern",
]

# B・Cの問題用データは「連番・同じ文字・連番と同じ文字・ランダム」
# の順に、1:1:1:7の割合で選ぶ
BC_TEST_PATTERN_COUNTS = {
    "sequence": 20,
    "repeated_character": 20,
    "sequence_and_repeated": 20,
    "random": 140,
}

# 同じ設定で実行すれば、毎回同じデータを選びやすくする番号
RANDOM_STATE = 42


# --------------------------------------------------
# AI作成準備で使う部品
# CSVを読み込み、保存先を作り、学習用と問題用に分ける
# --------------------------------------------------

# --------------------------------------------------
# 1-1. モデルの保存先フォルダーを用意する
# --------------------------------------------------

def create_model_folder():
    """モデルやグラフを保存するフォルダーを作る。"""

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1-2. CSVファイルを読み込む
# --------------------------------------------------

def load_data():
    """英語の列名が付いたCSVを読み込む。"""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"CSVが見つかりません: {DATA_FILE}\n"
            "dataフォルダーにpassword_dataset.csvを入れてください。"
        )

    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    needed_columns = AI_INPUT_COLUMNS + [
        "password",
        "rank",
        "pattern_type",
    ]
    missing_columns = [
        column for column in needed_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSVに必要な列がありません: " + ", ".join(missing_columns)
        )

    # AIに使う数字と、正解ランク・データ種類の列を取り出す
    data = data[needed_columns].dropna().copy()
    data["rank"] = data["rank"].astype(str).str.strip().str.upper()
    data["pattern_type"] = (
        data["pattern_type"].astype(str).str.strip().str.lower()
    )

    # 生成コードの版による種類名の違いを、学習コード内の名前にそろえる。
    data["pattern_type"] = data["pattern_type"].replace({
        "repeated_pair": "repeated_character",
        "sequence_and_repeated_pair": "sequence_and_repeated",
    })

    data = data[data["rank"].isin(RANKS)]

    # 数字の列を、AIが読み込める数値にそろえる
    for column in AI_INPUT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=AI_INPUT_COLUMNS)

    for rank in RANKS:
        count = int((data["rank"] == rank).sum())
        expected_count = TRAIN_COUNT_PER_RANK + TEST_COUNT_PER_RANK

        if count != expected_count:
            raise ValueError(
                f"{rank}ランクは{expected_count}件必要ですが、"
                f"{count}件ありました。"
            )

    return data


# --------------------------------------------------
# 1-3. 学習用と問題用に分ける
# --------------------------------------------------

def split_data(data):
    """各ランクから学習用1,800件と問題用200件を分ける。"""

    training_parts = []
    test_parts = []

    for rank in RANKS:
        rank_data = data[data["rank"] == rank].copy()

        if rank == "A":
            # Aはランダムデータから問題用を200件選ぶ
            test_data = rank_data[
                rank_data["pattern_type"] == "random"
            ].sample(
                n=TEST_COUNT_PER_RANK,
                random_state=RANDOM_STATE,
            )
        else:
            # B・Cは指定した4種類から、1:1:1:7の数で選ぶ
            selected_tests = []

            for pattern, count in BC_TEST_PATTERN_COUNTS.items():
                pattern_data = rank_data[
                    rank_data["pattern_type"] == pattern
                ]

                if len(pattern_data) < count:
                    available_patterns = rank_data["pattern_type"].value_counts()
                    available_text = ", ".join(
                        f"{name}: {number}件"
                        for name, number in available_patterns.items()
                    )
                    raise ValueError(
                        f"{rank}ランクの{pattern}が問題用に足りません。"
                        f"必要数: {count}件、現在の種類別件数: {available_text}"
                    )

                selected_tests.append(
                    pattern_data.sample(
                        n=count,
                        random_state=RANDOM_STATE,
                    )
                )

            test_data = pd.concat(selected_tests)

        # 問題用にしなかったデータは、すべて学習用にする。
        # 年と月日のデータは問題用に選ばないため、全部学習に使われる。
        training_data = rank_data.drop(index=test_data.index)

        if len(training_data) != TRAIN_COUNT_PER_RANK:
            raise ValueError(
                f"{rank}ランクの学習用データが"
                f"{len(training_data)}件です。"
            )

        training_parts.append(training_data)
        test_parts.append(test_data)

    training_data = pd.concat(training_parts).sample(
        frac=1,
        random_state=RANDOM_STATE,
    )
    test_data = pd.concat(test_parts).sample(
        frac=1,
        random_state=RANDOM_STATE,
    )

    return training_data, test_data


# --------------------------------------------------
# 1-4. AI作成準備をまとめる
# --------------------------------------------------

def prepare_ai_data():
    """CSVを読み、保存先を作り、学習用と問題用に分ける。"""

    data = load_data()

    # AIを作る前に、モデルとグラフの保存先を用意する
    create_model_folder()

    training_data, test_data = split_data(data)

    return data, training_data, test_data


# ==================================================
# 2. AI作成
# 学習用データで学習し、予測モデルを保存する
# ==================================================

# --------------------------------------------------
# 2-1. ランダムフォレストの枠組みを作る
# --------------------------------------------------

def create_ai_frame():
    """学習前のランダムフォレストを用意する。"""

    return RandomForestClassifier(
        n_estimators=TREE_COUNT,
        random_state=RANDOM_STATE,
    )


# --------------------------------------------------
# 2-2. 学習用データをAIに読み込ませる
# --------------------------------------------------

def train_ai(ai, training_numbers, training_answers):
    """学習用の数字と正解ランクをAIに教える。"""

    ai.fit(training_numbers, training_answers)

    return ai


# --------------------------------------------------
# 2-3. 学習済みAIを保存する
# --------------------------------------------------

def save_ai(ai):
    """学習済みAIを、設定した保存先に保存する。"""

    joblib.dump(
        {
            "model": ai,
            "input_columns": AI_INPUT_COLUMNS,
        },
        MODEL_FILE,
    )


# --------------------------------------------------
# 2-4. AI作成・学習・保存をひとまとめにする
# --------------------------------------------------

def create_and_save_model(training_data):
    """AIを作り、学習用データで学習させて、モデルを保存する。"""

    training_numbers = training_data[AI_INPUT_COLUMNS]
    training_answers = training_data["rank"]

    ai = create_ai_frame()
    ai = train_ai(ai, training_numbers, training_answers)
    save_ai(ai)

    return ai


# ==================================================
# 3. 学習結果の表とグラフを作る
# ==================================================

# --------------------------------------------------
# 3-1. 正答率をpandasの表にまとめる
# --------------------------------------------------

def make_accuracy_table(test_answers, predictions):
    """ランクごとの正解数と正答率を表にする。"""

    rows = []

    for rank in RANKS:
        rank_answers = test_answers == rank
        total_count = int(rank_answers.sum())
        correct_count = int(
            (predictions[rank_answers] == test_answers[rank_answers]).sum()
        )
        accuracy = correct_count / total_count * 100

        rows.append({
            "ランク": rank,
            "正解数": correct_count,
            "問題数": total_count,
            "正答率（％）": accuracy,
        })

    total_correct = sum(row["正解数"] for row in rows)
    total_questions = sum(row["問題数"] for row in rows)

    rows.append({
        "ランク": "合計",
        "正解数": total_correct,
        "問題数": total_questions,
        "正答率（％）": total_correct / total_questions * 100,
    })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 3-2. 学習データの表を表示する
# --------------------------------------------------

def show_data_table(data):
    """読み込んだCSVの一部を、表として画面に表示する。"""

    print("\nCSVから読み込んだ表の例（最初の5行）:")
    print(
        data[AI_INPUT_COLUMNS + ["rank", "pattern_type"]]
        .head()
        .to_string(index=False)
    )


# --------------------------------------------------
# 3-3. 正答率の表を表示する
# --------------------------------------------------

def show_accuracy_table(accuracy_table):
    """正答率をpandasの表として画面に表示する。"""

    print("\n問題の正答率をまとめた表:")
    print(accuracy_table.to_string(index=False, formatters={
        "正答率（％）": "{:.1f}%".format,
    }))


# --------------------------------------------------
# 3-4. 正答率のグラフを作って保存する
# --------------------------------------------------

def save_accuracy_graph(accuracy_table):
    """正答率の表から、A・B・Cの棒グラフを作る。"""

    graph_data = accuracy_table[
        accuracy_table["ランク"] != "合計"
    ]

    # Windowsにある日本語フォントを使う
    plt.rcParams["font.family"] = "Yu Gothic"

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        graph_data["ランク"],
        graph_data["正答率（％）"],
        color=["#4C78A8", "#F2A541", "#E45756"],
    )

    plt.title("ランクごとの正答率")
    plt.xlabel("ランク")
    plt.ylabel("正答率（％）")

    # 100%の数字がグラフからはみ出さないように余白を作る
    plt.ylim(0, 110)
    plt.yticks(range(0, 101, 20))
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)
    plt.close()


# --------------------------------------------------
# 3-5. 表とグラフをまとめて作る
# --------------------------------------------------

def create_result_report(ai, data, training_data, test_data):
    """問題を判定し、表とグラフを作って結果を表示する。"""

    test_numbers = test_data[AI_INPUT_COLUMNS]
    test_answers = test_data["rank"]
    predictions = ai.predict(test_numbers)
    accuracy_table = make_accuracy_table(test_answers, predictions)

    show_data_table(data)
    show_accuracy_table(accuracy_table)
    save_accuracy_graph(accuracy_table)

    print("\nAIの作成が終わりました。")
    print(f"学習に使った数: {len(training_data)}件（各ランク1,800件）")
    print(f"問題に使った数: {len(test_data)}件（各ランク200件）")
    print(f"モデルの保存先: {MODEL_FILE}")
    print(f"正答率グラフの保存先: {GRAPH_FILE}")


# ==================================================
# 4. 用意した「ボタン」を順番に実行する
# ==================================================

def main():
    """各章で用意した処理を、上から順番に呼び出す。"""

    # ボタン1: AI作成準備。CSVを読み、学習用と問題用に分ける
    data, training_data, test_data = prepare_ai_data()

    # ボタン2: AI作成。学習して、予測モデルを保存する
    ai = create_and_save_model(training_data)

    # ボタン3: 問題を判定して、表とグラフを作る
    create_result_report(ai, data, training_data, test_data)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("AIを作成できませんでした。")
        print(f"理由: {error}")
