"""
パスワードのサンプルを作って、CSVに保存する。
機械学習用は英語の列名、確認用は日本語の列名にする。
"""

import csv
import datetime
import re
import secrets
import string
from pathlib import Path

import pandas as pd


# ==================================================
# 1. 設定
# ==================================================

# --------------------------------------------------
# その他の設定
# --------------------------------------------------

# ランクごとに作るデータの件数
DATA_COUNT_PER_RANK = 2000

# B・Cランクのデータ件数
# 作り方ごとの件数を足すと、それぞれ2,000件になる
PATTERN_COUNTS_BY_RANK = {
    "B": {
        "random": 522,
        "sequence": 400,
        "repeated_pair": 400,
        "sequence_and_repeated_pair": 200,
        "year": 125,
        "date": 353,
    },
    "C": {
        "random": 322,
        "sequence": 400,
        "repeated_pair": 400,
        "sequence_and_repeated_pair": 200,
        "password_word": 100,
        "keyboard_sequence": 100,
        "year": 125,
        "date": 353,
    },
}

# 「連番＋同じ文字の連続」のパスワードは20文字まで
MAX_COMBINED_PATTERN_LENGTH = 20

# 年の範囲
MIN_YEAR = 1900
MAX_YEAR = 2026

# CSVファイルを保存するフォルダ
SAVE_FOLDER = Path("data")

# パスワードに使う文字
LOWERCASE = string.ascii_lowercase
UPPERCASE = string.ascii_uppercase
NUMBERS = string.digits
SYMBOLS = "!@#$%&*?_-+"
ALL_CHARACTERS = LOWERCASE + UPPERCASE + NUMBERS + SYMBOLS

# キーボードの横一列の並び
KEYBOARD_ROWS = (
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "1234567890",
)

# よく使われる単語
COMMON_WORDS = (
    "password",
    "hello",
)

# 機械学習用CSVの列名
CSV_COLUMNS = [
    "password_id",
    "password",
    "length",
    "has_sequence",
    "has_repeated_pair",
    "has_weak_pattern",
    "rank",
    "pattern_type",
]

# 作成済みのパスワードを覚えておき、重複を防ぐ
used_passwords = set()


# --------------------------------------------------
# 絶対条件
# ひとつでも満たさないパスワードは使わない
# --------------------------------------------------

ABSOLUTE_CONDITIONS = {
    "min_length": 7,                    # 最低7文字
    "require_lowercase": True,          # 小文字を入れる
    "require_uppercase": True,          # 大文字を入れる
    "require_number": True,             # 数字を入れる
    "require_symbol": True,             # 記号を入れる
    "max_consecutive_identical": 2,     # 同じ文字は最大2回まで続けてよい
}


# --------------------------------------------------
# 評価条件
# 特徴があるかを調べて、CSVに1（あり）か0（なし）で記録する
# --------------------------------------------------

EVALUATION_CONDITIONS = {
    "sequence_length": 3,               # 3文字以上の連番を調べる
    "repeated_character_count": 2,      # 同じ文字が2回以上続くか調べる
}


# --------------------------------------------------
# ランク設定
# Cは7～9文字、Aは16～19文字
# 通常のBは10～15文字、弱いパターンのない20～24文字もB
# 16文字は連番があればB
# 17～19文字は連番と同じ文字の連続が両方あればB
# --------------------------------------------------

RANK_RULES = {
    "C": {"min_length": 7, "max_length": 9},
    "B": {
        "min_length": 10,
        "max_length": 15,
        "long_min_length": 20,
        "long_max_length": 24,
    },
    "A": {"min_length": 16, "max_length": 19},
}


# ==================================================
# 2. パスワードを作る部品
# ここでは使う関数を用意して、最後にまとめて実行する
# ==================================================


# --------------------------------------------------
# 2-1. 絶対条件に合っているか調べる
# --------------------------------------------------

def has_repeated_run(password, count):
    """同じ文字がcount回以上続いているか調べる。"""

    pattern = rf"(.)\1{{{count - 1},}}"
    return re.search(pattern, password) is not None


def meets_basic_requirements(password):
    """絶対条件をすべて満たしていればTrueを返す。"""

    rules = ABSOLUTE_CONDITIONS

    # 最低文字数に足りているか
    if len(password) < rules["min_length"]:
        return False

    # 小文字が入っているか
    if rules["require_lowercase"] and not any(
        char in LOWERCASE for char in password
    ):
        return False

    # 大文字が入っているか
    if rules["require_uppercase"] and not any(
        char in UPPERCASE for char in password
    ):
        return False

    # 数字が入っているか
    if rules["require_number"] and not any(
        char in NUMBERS for char in password
    ):
        return False

    # 記号が入っているか
    if rules["require_symbol"] and not any(
        char in SYMBOLS for char in password
    ):
        return False

    # 同じ文字が3回以上続いたら使わない
    max_count = rules["max_consecutive_identical"]

    if has_repeated_run(password, max_count + 1):
        return False

    return True


# --------------------------------------------------
# 2-2. 評価条件に当てはまるか調べる
# --------------------------------------------------

def has_sequence(password):
    """
    3文字以上の連番があるか調べる。
    連番があれば1、なければ0を返す。
    例：abc、XYZ、123、987
    """

    sequence_length = EVALUATION_CONDITIONS["sequence_length"]
    lowered_password = password.lower()

    sequences = (
        string.ascii_lowercase,
        string.ascii_lowercase[::-1],
        string.digits,
        string.digits[::-1],
    )

    for index in range(len(lowered_password) - sequence_length + 1):
        part = lowered_password[index:index + sequence_length]

        if any(part in sequence for sequence in sequences):
            return 1

    return 0


def has_repeated_pair(password):
    """
    同じ文字が2回以上続いているか調べる。
    当てはまれば1、なければ0を返す。
    """

    repeat_count = EVALUATION_CONDITIONS["repeated_character_count"]
    return int(has_repeated_run(password, repeat_count))


def has_keyboard_sequence(password):
    """キーボードの並びが3文字以上入っているか調べる。"""

    lowered_password = password.lower()

    for row in KEYBOARD_ROWS:
        # 3文字、4文字……と少しずつ切り出して探す
        for length in range(3, len(row) + 1):
            for start in range(len(row) - length + 1):
                part = row[start:start + length]

                if part in lowered_password or part[::-1] in lowered_password:
                    return True

    return False


def has_common_word(password):
    """passwordやhelloなど、指定した単語が入っているか調べる。"""

    lowered_password = password.lower()
    return any(word in lowered_password for word in COMMON_WORDS)


def has_year(password):
    """1900～2026の年が、単独の4桁として入っているか調べる。"""

    year_pattern = (
        r"(?<!\d)(?:19\d{2}|20(?:0\d|1\d|2[0-6]))(?!\d)"
    )
    return re.search(year_pattern, password) is not None


def is_valid_month_day(month, day):
    """月日が実際にある日付か調べる。"""

    try:
        # うるう年を含む2000年で、2月29日も確認する
        datetime.date(2000, month, day)
        return True
    except ValueError:
        return False


def has_month_day(password):
    """0101～1231のような月日が入っているか調べる。"""

    month_day_pattern = r"(?<!\d)(\d{2})(\d{2})(?!\d)"

    for match in re.finditer(month_day_pattern, password):
        month = int(match.group(1))
        day = int(match.group(2))

        if is_valid_month_day(month, day):
            return True

    return False


def has_weak_pattern(password):
    """
    キーボードの並び、単語、年、日付などがあるか調べる。
    どれかがあれば1、なければ0を返す。
    """

    if has_keyboard_sequence(password):
        return 1

    if has_common_word(password):
        return 1

    if has_year(password):
        return 1

    if has_month_day(password):
        return 1

    return 0


# --------------------------------------------------
# 2-3. パスワードのランクを決める
# --------------------------------------------------

def determine_rank(password, generation_rank=None, pattern_type=None):
    """
    文字数と特徴からランクを決める。

    弱いパターンは基本的にCにする。
    ただし、B用に作った16文字以上の年・日付データはBにする。
    """

    password_length = len(password)

    # B用の年・日付データには、今回決めた例外を使う
    if (
        generation_rank == "B"
        and password_length >= 16
        and (
            (pattern_type == "year" and has_year(password))
            or (pattern_type == "date" and has_month_day(password))
        )
    ):
        return "B"

    # それ以外で弱いパターンがあれば、文字数に関係なくC
    if has_weak_pattern(password):
        return "C"

    # 7～9文字ならC
    if (
        RANK_RULES["C"]["min_length"]
        <= password_length
        <= RANK_RULES["C"]["max_length"]
    ):
        return "C"

    # 10～15文字ならB
    if (
        RANK_RULES["B"]["min_length"]
        <= password_length
        <= RANK_RULES["B"]["max_length"]
    ):
        return "B"

    sequence_exists = bool(has_sequence(password))
    repeated_pair_exists = bool(has_repeated_pair(password))

    # 16文字は、連番があればB
    if password_length == 16:
        if sequence_exists:
            return "B"
        return "A"

    # 17～19文字は、連番と同じ文字の連続が両方あればB
    if (
        RANK_RULES["A"]["min_length"]
        <= password_length
        <= RANK_RULES["A"]["max_length"]
    ):
        if sequence_exists and repeated_pair_exists:
            return "B"
        return "A"

    # 20～24文字は、弱いパターンがなければB
    if (
        RANK_RULES["B"]["long_min_length"]
        <= password_length
        <= RANK_RULES["B"]["long_max_length"]
    ):
        return "B"

    return None


# --------------------------------------------------
# 2-4. パスワードに入れる連番を作る
# --------------------------------------------------

def generate_sequence(password_length):
    """小文字・大文字・数字の中から3～5文字の連番を作る。"""

    max_sequence_length = min(5, password_length)
    sequence_length = secrets.randbelow(max_sequence_length - 2) + 3

    # 小文字・大文字・数字のどれかを選ぶ
    characters = secrets.choice((LOWERCASE, UPPERCASE, NUMBERS))

    # 連番をどこから始めるか決める
    start_index = secrets.randbelow(
        len(characters) - sequence_length + 1
    )

    sequence = characters[start_index:start_index + sequence_length]

    # 半分くらいの確率で逆向きにする
    if secrets.randbelow(2):
        sequence = sequence[::-1]

    return sequence


# --------------------------------------------------
# 2-5. 年や日付の数字を作る
# --------------------------------------------------

def generate_year():
    """1900～2026から、同じ数字が3回続かない年を作る。"""

    available_years = get_all_year_patterns()
    return secrets.choice(available_years)


def get_all_year_patterns():
    """条件に合う1900～2026の年をすべて返す。"""

    return [
        str(year)
        for year in range(MIN_YEAR, MAX_YEAR + 1)
        if not has_repeated_run(
            str(year),
            ABSOLUTE_CONDITIONS["max_consecutive_identical"] + 1,
        )
    ]


def get_all_month_day_patterns():
    """3文字連続にならない、実在する月日をすべて返す。"""

    start_date = datetime.date(2000, 1, 1)
    end_date = datetime.date(2001, 1, 1)
    number_of_days = (end_date - start_date).days

    return [
        month_day
        for offset in range(number_of_days)
        for month_day in [
            (start_date + datetime.timedelta(days=offset)).strftime("%m%d")
        ]
        if not has_repeated_run(
            month_day,
            ABSOLUTE_CONDITIONS["max_consecutive_identical"] + 1,
        )
    ]


def generate_month_day():
    """3文字連続にならない実在の月日を作る。"""

    return secrets.choice(get_all_month_day_patterns())


# --------------------------------------------------
# 2-6. パスワードをひとつ作る
# --------------------------------------------------

def generate_password(password_length, pattern_type, pattern_value=None):
    """指定した文字数と作り方でパスワードを作る。"""

    # 小文字・大文字・数字・記号をひとつずつ入れておく
    characters = [
        secrets.choice(LOWERCASE),
        secrets.choice(UPPERCASE),
        secrets.choice(NUMBERS),
        secrets.choice(SYMBOLS),
    ]

    # 指定した文字数になるまで使える文字を追加する
    while len(characters) < password_length:
        characters.append(secrets.choice(ALL_CHARACTERS))

    # 文字の順番をばらばらにする
    secrets.SystemRandom().shuffle(characters)
    password = list(characters)

    # 入れたい文字の並びを決める
    if pattern_type == "sequence":
        pattern = generate_sequence(password_length)

    elif pattern_type == "sequence_and_repeated_pair":
        pattern = generate_sequence(password_length)

    elif pattern_type == "password_word":
        # パスワードの長さに収まる単語だけを選ぶ
        available_words = [
            word for word in COMMON_WORDS
            if len(word) <= password_length
        ]

        # 入る単語がない場合は作り直す
        if not available_words:
            return ""

        pattern = secrets.choice(available_words)

    elif pattern_type == "keyboard_sequence":
        row = secrets.choice(KEYBOARD_ROWS)
        pattern_length = secrets.choice((3, 4, 5))
        start = secrets.randbelow(len(row) - pattern_length + 1)
        pattern = row[start:start + pattern_length]

        if secrets.randbelow(2):
            pattern = pattern[::-1]

    elif pattern_type == "year":
        pattern = pattern_value or generate_year()

    elif pattern_type == "date":
        pattern = pattern_value or generate_month_day()

    else:
        pattern = ""

    # 入れたい文字の並びをパスワードに入れる
    if pattern:
        # パターンが長すぎると配置できないため、作り直す
        if len(pattern) > password_length:
            return ""

        start_index = secrets.randbelow(password_length - len(pattern) + 1)
        password[start_index:start_index + len(pattern)] = list(pattern)

    # 連番と同じ文字を両方入れるパターン
    if pattern_type in ("repeated_pair", "sequence_and_repeated_pair"):
        pair_positions = []

        if pattern_type == "sequence_and_repeated_pair":
            sequence_start = start_index
            sequence_end = sequence_start + len(pattern) - 1

            # 連番と重ならない場所を探す
            for pair_start in range(password_length - 1):
                pair_end = pair_start + 1

                overlaps_sequence = (
                    pair_start <= sequence_end
                    and pair_end >= sequence_start
                )

                if not overlaps_sequence:
                    pair_positions.append(pair_start)
        else:
            pair_positions = list(range(password_length - 1))

        if not pair_positions:
            return ""

        pair_start = secrets.choice(pair_positions)
        repeated_char = secrets.choice(ALL_CHARACTERS)

        password[pair_start:pair_start + 2] = [
            repeated_char,
            repeated_char,
        ]

    return "".join(password)


# --------------------------------------------------
# 2-7. ランクと作り方に合わせて文字数を決める
# --------------------------------------------------

def choose_length_and_pattern(rank, pattern_type):
    """ランクと作り方に合わせて、パスワードの文字数を選ぶ。"""

    # Bランクは作り方に合わせて文字数の範囲を選ぶ
    if rank == "B":
        if pattern_type == "sequence_and_repeated_pair":
            length_ranges = [
                (10, MAX_COMBINED_PATTERN_LENGTH),
            ]
        elif pattern_type == "sequence":
            length_ranges = [
                (10, 16),
                (20, 24),
            ]
        elif pattern_type == "repeated_pair":
            length_ranges = [
                (10, 15),
                (20, 24),
            ]
        elif pattern_type in ("year", "date"):
            length_ranges = [
                (16, 24),
            ]
        else:
            length_ranges = [
                (10, 15),
                (20, 24),
            ]

        min_length, max_length = secrets.choice(length_ranges)

    # Cランクは7～9文字
    elif rank == "C":
        min_length = RANK_RULES["C"]["min_length"]
        max_length = RANK_RULES["C"]["max_length"]

    # Aランクは16～19文字
    else:
        min_length = RANK_RULES["A"]["min_length"]
        max_length = RANK_RULES["A"]["max_length"]

    password_length = (
        secrets.randbelow(max_length - min_length + 1)
        + min_length
    )

    return password_length, pattern_type


# --------------------------------------------------
# 2-8. 作り方と実際の特徴が合っているか調べる
# --------------------------------------------------

def matches_pattern(password, pattern_type):
    """選んだ作り方と、実際に入っている並びが合っているか調べる。"""

    sequence_exists = bool(has_sequence(password))
    repeated_pair_exists = bool(has_repeated_pair(password))

    if pattern_type == "random":
        return (
            not sequence_exists
            and not repeated_pair_exists
            and not has_weak_pattern(password)
        )

    if pattern_type == "sequence":
        return sequence_exists and not repeated_pair_exists

    if pattern_type == "repeated_pair":
        return repeated_pair_exists and not sequence_exists

    if pattern_type == "sequence_and_repeated_pair":
        return sequence_exists and repeated_pair_exists

    if pattern_type == "password_word":
        return has_common_word(password)

    if pattern_type == "keyboard_sequence":
        return has_keyboard_sequence(password)

    if pattern_type == "year":
        return has_year(password)

    if pattern_type == "date":
        return has_month_day(password)

    return False


# --------------------------------------------------
# 2-9. 指定ランクと作り方のデータを作る
# --------------------------------------------------

def generate_rank_data(rank, data_count, pattern_type):
    """指定したランクと作り方で、必要な件数のデータを作る。"""

    records = []

    # 年と日付は、条件に合う値をひとつずつすべて使う
    if pattern_type == "year":
        pattern_values = get_all_year_patterns()
    elif pattern_type == "date":
        pattern_values = get_all_month_day_patterns()
    else:
        pattern_values = [None] * data_count

    if len(pattern_values) != data_count:
        raise ValueError(
            f"{rank}ランクの「{pattern_type}」は"
            f"{len(pattern_values)}件に設定してください。"
        )

    for pattern_value in pattern_values:
        attempts = 0

        while True:
            attempts += 1

            # 何度試しても作れないときに、処理が止まらなくなるのを防ぐ
            if attempts > 3000:
                raise RuntimeError(
                    f"{rank}ランクの「{pattern_type}」データを"
                    "必要数作成できませんでした。"
                )

            # 今回の文字数を決めて、パスワードを作る
            password_length, _ = choose_length_and_pattern(
                rank,
                pattern_type,
            )
            password = generate_password(
                password_length,
                pattern_type,
                pattern_value,
            )

            # うまく作れなかった場合は、もう一度試す
            if not password:
                continue

            # 絶対条件に合わないものは作り直す
            if not meets_basic_requirements(password):
                continue

            # すでに使ったパスワードなら作り直す
            if password in used_passwords:
                continue

            # 作り方と実際の特徴が合っているか調べる
            if not matches_pattern(password, pattern_type):
                continue

            # 実際のランクが指定したランクと違えば作り直す
            actual_rank = determine_rank(
                password,
                generation_rank=rank,
                pattern_type=pattern_type,
            )

            if actual_rank != rank:
                continue

            # 採用したパスワードを覚えておき、記録に追加する
            used_passwords.add(password)

            records.append({
                "password": password,
                "length": len(password),
                "has_sequence": has_sequence(password),
                "has_repeated_pair": has_repeated_pair(password),
                "has_weak_pattern": has_weak_pattern(password),
                "rank": actual_rank,
                "pattern_type": pattern_type,
            })
            break

    return records


# ==================================================
# 3. CSVに保存する部品
# ==================================================


# --------------------------------------------------
# 3-1. CSVファイルをひとつ保存する
# --------------------------------------------------

def save_one_csv(file_path, records):
    """指定した場所にCSVファイルを保存する。"""

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(records)


# --------------------------------------------------
# 3-2. ランク別・全件・日本語版のCSVを保存する
# --------------------------------------------------

def save_all_csv(rank_data):
    """ランク別CSV、機械学習用CSV、日本語版CSVを保存する。"""

    all_records = []

    # 作り方を日本語で表示するときの名前
    pattern_names = {
        "random": "ランダム",
        "sequence": "連番",
        "repeated_pair": "同じ文字の2回連続",
        "sequence_and_repeated_pair": "連番と同じ文字の2回連続",
        "password_word": "password・hello",
        "keyboard_sequence": "キーボードの並び",
        "year": "年",
        "date": "月日",
    }

    # A・B・CそれぞれのCSVを作る
    for rank in ("A", "B", "C"):
        rank_records = []

        for number, record in enumerate(rank_data[rank], start=1):
            row = {
                "password_id": f"{rank}-{number:04d}",
                **record,
            }
            rank_records.append(row)

        # ランク別の保存先フォルダを作る
        rank_folder = SAVE_FOLDER / rank
        rank_folder.mkdir(parents=True, exist_ok=True)

        # ランク別CSVを保存する
        rank_file = rank_folder / f"passwords_{rank}.csv"
        save_one_csv(rank_file, rank_records)

        print(f"保存しました: {rank_file}")
        all_records.extend(rank_records)

    # 全ランクをまとめた機械学習用CSVを保存する
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    all_file = SAVE_FOLDER / "password_dataset.csv"
    save_one_csv(all_file, all_records)
    print(f"保存しました: {all_file}")

    # 日本語の列名で確認できるデータを作る
    japanese_records = []

    for record in all_records:
        japanese_records.append({
            "パスワードID": record["password_id"],
            "パスワード": record["password"],
            "文字数": record["length"],
            "連番あり": record["has_sequence"],
            "同じ文字の2回連続あり": record["has_repeated_pair"],
            "弱いパターンあり": record["has_weak_pattern"],
            "ランク": record["rank"],
            "作成パターン": pattern_names.get(
                record["pattern_type"],
                record["pattern_type"],
            ),
        })

    # 日本語版CSVを保存する
    japanese_file = SAVE_FOLDER / "password_dataset_japanese.csv"

    pd.DataFrame(japanese_records).to_csv(
        japanese_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"保存しました: {japanese_file}")
    print(f"合計 {len(all_records)} 件のデータを保存しました。")


# ==================================================
# 4. 件数表示と、最後にまとめて実行する部分
# ==================================================


# --------------------------------------------------
# 4-1. ランクごとの特徴の件数を表示する
# --------------------------------------------------

def show_pattern_counts(rank_data):
    """ランクごとに、作り方や特徴が何件あるか表示する。"""

    print("\n【ランク別の件数】")

    for rank in ("A", "B", "C"):
        df = pd.DataFrame(rank_data[rank])

        print(f"\n【ランク{rank}】")
        print(f"総件数：{len(df)}件")
        print(f"連番あり：{df['has_sequence'].sum()}件")
        print(
            "同じ文字の連続あり："
            f"{df['has_repeated_pair'].sum()}件"
        )
        print(f"弱いパターンあり：{df['has_weak_pattern'].sum()}件")
        print("\n作り方ごとの件数：")
        print(df["pattern_type"].value_counts())


# --------------------------------------------------
# 4-2. 用意した部品を使って、最後にまとめて動かす
# --------------------------------------------------

def main():
    """A・B・Cのデータを作り、CSVに保存する。"""

    print("パスワードデータの生成を開始します。")
    print(
        "同じ文字が"
        f"{ABSOLUTE_CONDITIONS['max_consecutive_identical'] + 1}回以上"
        "続くパスワードは使いません。"
    )

    rank_data = {}

    # Aはランダムなデータを2,000件作る
    print("\nAランクのデータを作成中...")
    rank_data["A"] = generate_rank_data(
        "A",
        DATA_COUNT_PER_RANK,
        "random",
    )

    # B・Cは設定した内訳で、それぞれ2,000件作る
    for rank in ("B", "C"):
        print(f"\n{rank}ランクのデータを作成中...")
        rank_data[rank] = []

        for pattern_type, count in PATTERN_COUNTS_BY_RANK[rank].items():
            print(f"  {pattern_type}：{count}件")

            rank_data[rank].extend(
                generate_rank_data(
                    rank,
                    count,
                    pattern_type,
                )
            )

    # どんなデータができたか表示する
    show_pattern_counts(rank_data)

    # 作ったデータをCSVに保存する
    save_all_csv(rank_data)

    print("\nすべて完了しました。")


# このファイルを実行したときに、最後の処理を始める
if __name__ == "__main__":
    main()
