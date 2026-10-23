"""保存済みAIでパスワードを判定し、20～24文字を生成するアプリ。"""

import datetime
import re
import secrets
import string
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import joblib


# 作成済みAIの場所
APP_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = APP_FOLDER / "model" / "password_strength_model.joblib"

# AI作成プログラムで使った入力列
# AIはパスワードの文字を直接読むのではなく、この4つの数字で学習している。
# 入力パスワードから同じ数字を作って、保存済みモデルに渡す。
INPUT_COLUMNS = [
    "length",
    "has_sequence",
    "has_repeated_pair",
    "has_weak_pattern",
]

# パスワード生成に使う文字
LOWERCASE = string.ascii_lowercase
UPPERCASE = string.ascii_uppercase
DIGITS = string.digits
SYMBOLS = "!@#$%&*?_-+"
ALL_CHARACTERS = LOWERCASE + UPPERCASE + DIGITS + SYMBOLS
MIN_PASSWORD_LENGTH = 7


def has_sequence(password):
    """パスワードから、AI用の「連番あり・なし」を作る。"""

    # 大文字・小文字を区別せずに調べる。
    text = password.lower()

    # 英字と数字の並び。逆向きも入れて、cbaや321も見つける。
    rows = (
        string.ascii_lowercase,
        string.ascii_lowercase[::-1],
        string.digits,
        string.digits[::-1],
    )

    # 左から3文字ずつ見て、どれかの並びにあれば連番あり。
    return any(
        text[index:index + 3] in row
        for row in rows
        for index in range(len(text) - 2)
    )


def has_weak_pattern(password):
    """パスワードから、AI用の「弱いパターンあり・なし」を作る。"""

    # 大文字・小文字を区別せず、よく使われる単語を探す。
    text = password.lower()

    if "password" in text or "hello" in text:
        return True

    # キーボードの横並びを、順方向と逆方向の両方で調べる。
    keyboard_rows = (
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
        "1234567890",
    )

    for row in keyboard_rows:
        for size in range(3, len(row) + 1):
            for start in range(len(row) - size + 1):
                part = row[start:start + size]
                if part in text or part[::-1] in text:
                    return True

    # 1900～2026の年が、前後に別の数字を伴わずにあるか調べる。
    year = r"(?<!\d)(?:19\d{2}|20(?:0\d|1\d|2[0-6]))(?!\d)"
    if re.search(year, password):
        return True

    # 4桁の数字を月・日として読み、実際にある日付か調べる。
    for match in re.finditer(r"(?<!\d)(\d{2})(\d{2})(?!\d)", password):
        try:
            datetime.date(2000, int(match.group(1)), int(match.group(2)))
            return True
        except ValueError:
            pass

    return False


def load_model():
    """AI作成プログラムで保存したモデルを読み込む。"""

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"予測モデルが見つかりません。\n保存場所: {MODEL_FILE}"
        )

    # CSVを再学習するのではなく、保存済みのモデルファイルを開く。
    saved_data = joblib.load(MODEL_FILE)
    model = saved_data["model"]
    columns = saved_data["input_columns"]

    if columns != INPUT_COLUMNS:
        raise ValueError("予測モデルの入力項目が一致しません。")

    return model


def make_features(password):
    """入力パスワードを、学習時と同じ4つの数字に変える。"""

    # 項目の順番はINPUT_COLUMNSと同じ。ここで作った数字をAIが判定する。
    return [[
        len(password),
        int(has_sequence(password)),
        int(re.search(r"(.)\1", password) is not None),
        int(has_weak_pattern(password)),
    ]]


def check_absolute_conditions(password):
    """必ず守る条件を調べ、満たしていない項目を返す。"""

    problems = []

    if not password.isascii():
        problems.append("半角で入力する")
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"{MIN_PASSWORD_LENGTH}文字以上")
    if not any(char in LOWERCASE for char in password):
        problems.append("小文字を含む")
    if not any(char in UPPERCASE for char in password):
        problems.append("大文字を含む")
    if not any(char in DIGITS for char in password):
        problems.append("数字を含む")
    if not any(char in SYMBOLS for char in password):
        problems.append("記号を含む")
    if re.search(r"(.)\1{2}", password):
        problems.append("同じ文字を3回以上続けない")

    return problems


def make_password():
    """文字の種類をそろえた20～24文字のパスワードを作る。"""

    while True:
        length = secrets.choice(range(20, 25))
        characters = [
            secrets.choice(LOWERCASE),
            secrets.choice(UPPERCASE),
            secrets.choice(DIGITS),
            secrets.choice(SYMBOLS),
        ]

        while len(characters) < length:
            char = secrets.choice(ALL_CHARACTERS)

            # 同じ文字が3回続く場合は、別の文字を選び直す。
            if (
                len(characters) >= 2
                and characters[-1] == characters[-2] == char
            ):
                continue

            characters.append(char)

        secrets.SystemRandom().shuffle(characters)
        password = "".join(characters)

        # 並べ替えたあとも、同じ文字が3回続いていないか確認する。
        if not re.search(r"(.)\1{2}", password):
            return password


def main():
    """入力したパスワードの判定と、パスワード生成を行う。"""

    window = tk.Tk()
    window.title("パスワードランク判定")
    window.geometry("500x430")
    window.resizable(False, False)

    try:
        model = load_model()
        model_text = "予測モデルを読み込みました"
    except Exception as error:
        model = None
        model_text = f"モデルを読み込めません: {error}"

    tk.Label(
        window,
        text="パスワードランク判定",
        font=("Yu Gothic UI", 16, "bold"),
    ).pack(pady=(16, 4))

    tk.Label(
        window,
        text=model_text,
        wraplength=420,
        fg="#555555",
    ).pack(pady=(0, 12))

    tk.Label(
        window,
        text="パスワードを入力",
    ).pack(anchor="w", padx=24)

    password_row = tk.Frame(window)
    password_row.pack(padx=24, pady=5, fill="x")

    password_entry = tk.Entry(
        password_row,
        show="●",
        width=42,
    )
    password_entry.pack(side="left", fill="x", expand=True)

    def toggle_password():
        """目のボタンで入力パスワードの表示を切り替える。"""

        if password_entry.cget("show"):
            password_entry.config(show="")
        else:
            password_entry.config(show="●")

    tk.Button(
        password_row,
        text="👁",
        width=3,
        command=toggle_password,
    ).pack(side="left", padx=(6, 0))

    tk.Label(
        window,
        text=(
            "絶対条件：半角で入力／7文字以上／小文字・大文字・数字・記号を含む／"
            "同じ文字を3回以上続けない\n"
            "条件を満たさない入力は受け付けません。"
        ),
        justify="left",
        wraplength=450,
        fg="#555555",
    ).pack(anchor="w", padx=24, pady=(2, 6))

    result_label = tk.Label(
        window,
        text="判定結果: --",
        font=("Yu Gothic UI", 13, "bold"),
    )
    result_label.pack(pady=5)

    def check_password():
        """絶対条件を確認し、問題がなければ予測モデルに渡す。"""

        password = password_entry.get()

        if not password:
            result_label.config(text="パスワードを入力してください")
            return

        if model is None:
            result_label.config(text="予測モデルを読み込めません")
            return

        problems = check_absolute_conditions(password)

        if problems:
            # 絶対条件に合わない入力はAIに渡さず、理由を表示する。
            messagebox.showerror(
                "入力できません",
                "次の絶対条件を確認してください。\n・"
                + "\n・".join(problems),
                parent=window,
            )
            return

        # 条件を満たした入力だけを、読み込んだAIで判定する。
        rank = model.predict(make_features(password))[0]
        result_label.config(text=f"判定結果: {rank}ランク")

    tk.Button(
        window,
        text="ランクを判定",
        command=check_password,
        state="normal" if model is not None else "disabled",
    ).pack(pady=(0, 14))

    tk.Label(
        window,
        text="パスワードを生成（20～24文字）",
    ).pack(anchor="w", padx=24)

    generated_entry = tk.Entry(window, width=48)
    generated_entry.pack(padx=24, pady=5, fill="x")

    def generate_password():
        password = make_password()
        generated_entry.delete(0, tk.END)
        generated_entry.insert(0, password)

    tk.Button(
        window,
        text="パスワードを生成",
        command=generate_password,
    ).pack(pady=5)

    window.mainloop()


if __name__ == "__main__":
    main()
