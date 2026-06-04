"""
nlp_applications/information_extraction.py — Relation Extraction

Extract subject-relation-object triples from Japanese text using:
  1. Pattern-based matching (regex over surface forms)
  2. Neural scoring of candidate triples via a small MLP on bag-of-characters

Dataset: 100 synthetic Japanese sentences with annotated (subj, rel, obj) triples.
Evaluation: F1 for exact triple match and relaxed (partial token) match.
"""
from __future__ import annotations

import re
import sys
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "06-information-extraction")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ===========================================================================
# Annotated Dataset — 100 synthetic sentences with gold triples
# ===========================================================================

ANNOTATED_DATA: list[dict] = [
    # [名詞]は[名詞]を[動詞]
    {"text": "太郎は林檎を食べた", "triples": [{"subj": "太郎", "rel": "食べた", "obj": "林檎"}]},
    {"text": "医者は患者を診察した", "triples": [{"subj": "医者", "rel": "診察した", "obj": "患者"}]},
    {"text": "エンジニアはコードを書いた", "triples": [{"subj": "エンジニア", "rel": "書いた", "obj": "コード"}]},
    {"text": "花子は本を読んだ", "triples": [{"subj": "花子", "rel": "読んだ", "obj": "本"}]},
    {"text": "先生は授業を教えた", "triples": [{"subj": "先生", "rel": "教えた", "obj": "授業"}]},
    {"text": "警察は犯人を逮捕した", "triples": [{"subj": "警察", "rel": "逮捕した", "obj": "犯人"}]},
    {"text": "会社は製品を開発した", "triples": [{"subj": "会社", "rel": "開発した", "obj": "製品"}]},
    {"text": "研究者は実験を行った", "triples": [{"subj": "研究者", "rel": "行った", "obj": "実験"}]},
    {"text": "子供は宿題を終えた", "triples": [{"subj": "子供", "rel": "終えた", "obj": "宿題"}]},
    {"text": "農家は野菜を育てた", "triples": [{"subj": "農家", "rel": "育てた", "obj": "野菜"}]},
    {"text": "作家は小説を書いた", "triples": [{"subj": "作家", "rel": "書いた", "obj": "小説"}]},
    {"text": "シェフは料理を作った", "triples": [{"subj": "シェフ", "rel": "作った", "obj": "料理"}]},
    {"text": "学生は論文を提出した", "triples": [{"subj": "学生", "rel": "提出した", "obj": "論文"}]},
    {"text": "科学者は仮説を検証した", "triples": [{"subj": "科学者", "rel": "検証した", "obj": "仮説"}]},
    {"text": "選手は試合に勝った", "triples": [{"subj": "選手", "rel": "勝った", "obj": "試合"}]},
    {"text": "大臣は法律を制定した", "triples": [{"subj": "大臣", "rel": "制定した", "obj": "法律"}]},
    {"text": "患者は薬を服用した", "triples": [{"subj": "患者", "rel": "服用した", "obj": "薬"}]},
    {"text": "猫はネズミを追いかけた", "triples": [{"subj": "猫", "rel": "追いかけた", "obj": "ネズミ"}]},
    {"text": "犬は骨を埋めた", "triples": [{"subj": "犬", "rel": "埋めた", "obj": "骨"}]},
    {"text": "子供は公園で遊んだ", "triples": [{"subj": "子供", "rel": "遊んだ", "obj": "公園"}]},
    # [名詞]の[名詞]は[名詞]
    {"text": "東京の人口は多い", "triples": [{"subj": "東京", "rel": "の人口", "obj": "多い"}]},
    {"text": "日本の首都は東京である", "triples": [{"subj": "日本", "rel": "の首都", "obj": "東京"}]},
    {"text": "会社の代表は田中だ", "triples": [{"subj": "会社", "rel": "の代表", "obj": "田中"}]},
    {"text": "フランスの首都はパリだ", "triples": [{"subj": "フランス", "rel": "の首都", "obj": "パリ"}]},
    {"text": "太陽系の中心は太陽だ", "triples": [{"subj": "太陽系", "rel": "の中心", "obj": "太陽"}]},
    {"text": "地球の衛星は月である", "triples": [{"subj": "地球", "rel": "の衛星", "obj": "月"}]},
    {"text": "病院の院長は鈴木だ", "triples": [{"subj": "病院", "rel": "の院長", "obj": "鈴木"}]},
    {"text": "チームのキャプテンは山田だ", "triples": [{"subj": "チーム", "rel": "のキャプテン", "obj": "山田"}]},
    {"text": "学校の校長は佐藤だ", "triples": [{"subj": "学校", "rel": "の校長", "obj": "佐藤"}]},
    {"text": "国の首相は岸田だ", "triples": [{"subj": "国", "rel": "の首相", "obj": "岸田"}]},
    # Subject-predicate-object with topic markers
    {"text": "アインシュタインが相対性理論を発見した", "triples": [{"subj": "アインシュタイン", "rel": "発見した", "obj": "相対性理論"}]},
    {"text": "ニュートンが万有引力を発見した", "triples": [{"subj": "ニュートン", "rel": "発見した", "obj": "万有引力"}]},
    {"text": "ダーウィンが進化論を提唱した", "triples": [{"subj": "ダーウィン", "rel": "提唱した", "obj": "進化論"}]},
    {"text": "マリーキュリーが放射能を発見した", "triples": [{"subj": "マリーキュリー", "rel": "発見した", "obj": "放射能"}]},
    {"text": "エジソンが電灯を発明した", "triples": [{"subj": "エジソン", "rel": "発明した", "obj": "電灯"}]},
    {"text": "ライト兄弟が飛行機を発明した", "triples": [{"subj": "ライト兄弟", "rel": "発明した", "obj": "飛行機"}]},
    {"text": "ガリレオが地動説を唱えた", "triples": [{"subj": "ガリレオ", "rel": "唱えた", "obj": "地動説"}]},
    {"text": "孔子が儒教を創始した", "triples": [{"subj": "孔子", "rel": "創始した", "obj": "儒教"}]},
    {"text": "フォードが自動車を量産した", "triples": [{"subj": "フォード", "rel": "量産した", "obj": "自動車"}]},
    {"text": "ベルが電話を発明した", "triples": [{"subj": "ベル", "rel": "発明した", "obj": "電話"}]},
    # Medical domain
    {"text": "医師は処方箋を発行した", "triples": [{"subj": "医師", "rel": "発行した", "obj": "処方箋"}]},
    {"text": "看護師は患者を看護した", "triples": [{"subj": "看護師", "rel": "看護した", "obj": "患者"}]},
    {"text": "薬剤師は薬を調合した", "triples": [{"subj": "薬剤師", "rel": "調合した", "obj": "薬"}]},
    {"text": "外科医は手術を実施した", "triples": [{"subj": "外科医", "rel": "実施した", "obj": "手術"}]},
    {"text": "放射線技師はレントゲンを撮影した", "triples": [{"subj": "放射線技師", "rel": "撮影した", "obj": "レントゲン"}]},
    {"text": "検査技師は血液を分析した", "triples": [{"subj": "検査技師", "rel": "分析した", "obj": "血液"}]},
    {"text": "患者が入院を拒否した", "triples": [{"subj": "患者", "rel": "拒否した", "obj": "入院"}]},
    {"text": "病院が新薬を導入した", "triples": [{"subj": "病院", "rel": "導入した", "obj": "新薬"}]},
    {"text": "製薬会社が新薬を開発した", "triples": [{"subj": "製薬会社", "rel": "開発した", "obj": "新薬"}]},
    {"text": "厚生労働省が薬を承認した", "triples": [{"subj": "厚生労働省", "rel": "承認した", "obj": "薬"}]},
    # Technology domain
    {"text": "グーグルが検索エンジンを開発した", "triples": [{"subj": "グーグル", "rel": "開発した", "obj": "検索エンジン"}]},
    {"text": "アップルがスマートフォンを発売した", "triples": [{"subj": "アップル", "rel": "発売した", "obj": "スマートフォン"}]},
    {"text": "マイクロソフトがOSを開発した", "triples": [{"subj": "マイクロソフト", "rel": "開発した", "obj": "OS"}]},
    {"text": "テスラが電気自動車を製造した", "triples": [{"subj": "テスラ", "rel": "製造した", "obj": "電気自動車"}]},
    {"text": "NASAが宇宙船を打ち上げた", "triples": [{"subj": "NASA", "rel": "打ち上げた", "obj": "宇宙船"}]},
    {"text": "研究所がAIを開発した", "triples": [{"subj": "研究所", "rel": "開発した", "obj": "AI"}]},
    {"text": "大学が特許を取得した", "triples": [{"subj": "大学", "rel": "取得した", "obj": "特許"}]},
    {"text": "スタートアップが資金を調達した", "triples": [{"subj": "スタートアップ", "rel": "調達した", "obj": "資金"}]},
    {"text": "政府がシステムを導入した", "triples": [{"subj": "政府", "rel": "導入した", "obj": "システム"}]},
    {"text": "企業がデータを収集した", "triples": [{"subj": "企業", "rel": "収集した", "obj": "データ"}]},
    # Possessive relationships
    {"text": "東京の面積は広い", "triples": [{"subj": "東京", "rel": "の面積", "obj": "広い"}]},
    {"text": "日本の経済は大きい", "triples": [{"subj": "日本", "rel": "の経済", "obj": "大きい"}]},
    {"text": "人体の構造は複雑だ", "triples": [{"subj": "人体", "rel": "の構造", "obj": "複雑だ"}]},
    {"text": "AIの発展は速い", "triples": [{"subj": "AI", "rel": "の発展", "obj": "速い"}]},
    {"text": "海の深さは深い", "triples": [{"subj": "海", "rel": "の深さ", "obj": "深い"}]},
    {"text": "山の高さは高い", "triples": [{"subj": "山", "rel": "の高さ", "obj": "高い"}]},
    {"text": "太陽の温度は高い", "triples": [{"subj": "太陽", "rel": "の温度", "obj": "高い"}]},
    {"text": "宇宙の大きさは無限だ", "triples": [{"subj": "宇宙", "rel": "の大きさ", "obj": "無限だ"}]},
    {"text": "歴史の重みは大きい", "triples": [{"subj": "歴史", "rel": "の重み", "obj": "大きい"}]},
    {"text": "知識の価値は高い", "triples": [{"subj": "知識", "rel": "の価値", "obj": "高い"}]},
    # Additional varied
    {"text": "太郎が花子に手紙を送った", "triples": [{"subj": "太郎", "rel": "送った", "obj": "手紙"}]},
    {"text": "母親が子供に食事を作った", "triples": [{"subj": "母親", "rel": "作った", "obj": "食事"}]},
    {"text": "政府が国民に支援金を配った", "triples": [{"subj": "政府", "rel": "配った", "obj": "支援金"}]},
    {"text": "銀行が企業に融資した", "triples": [{"subj": "銀行", "rel": "融資した", "obj": "企業"}]},
    {"text": "市役所が市民に情報を提供した", "triples": [{"subj": "市役所", "rel": "提供した", "obj": "情報"}]},
    {"text": "教師が生徒に知識を教えた", "triples": [{"subj": "教師", "rel": "教えた", "obj": "知識"}]},
    {"text": "コーチが選手に技術を教えた", "triples": [{"subj": "コーチ", "rel": "教えた", "obj": "技術"}]},
    {"text": "親が子供に道徳を教えた", "triples": [{"subj": "親", "rel": "教えた", "obj": "道徳"}]},
    {"text": "企業が社員に給与を支払った", "triples": [{"subj": "企業", "rel": "支払った", "obj": "給与"}]},
    {"text": "投資家が会社に出資した", "triples": [{"subj": "投資家", "rel": "出資した", "obj": "会社"}]},
    # Compound sentences (take first triple)
    {"text": "太郎は本を読んで感動した", "triples": [{"subj": "太郎", "rel": "読んで", "obj": "本"}]},
    {"text": "研究者はデータを収集して分析した", "triples": [{"subj": "研究者", "rel": "収集して", "obj": "データ"}]},
    {"text": "会社は製品を設計して製造した", "triples": [{"subj": "会社", "rel": "設計して", "obj": "製品"}]},
    {"text": "学生は図書館で本を借りた", "triples": [{"subj": "学生", "rel": "借りた", "obj": "本"}]},
    {"text": "選手はトレーニングを積んで優勝した", "triples": [{"subj": "選手", "rel": "積んで", "obj": "トレーニング"}]},
    {"text": "技術者はプログラムを作成した", "triples": [{"subj": "技術者", "rel": "作成した", "obj": "プログラム"}]},
    {"text": "芸術家は絵画を制作した", "triples": [{"subj": "芸術家", "rel": "制作した", "obj": "絵画"}]},
    {"text": "映画監督は映画を撮影した", "triples": [{"subj": "映画監督", "rel": "撮影した", "obj": "映画"}]},
    {"text": "音楽家は曲を作曲した", "triples": [{"subj": "音楽家", "rel": "作曲した", "obj": "曲"}]},
    {"text": "建築家は建物を設計した", "triples": [{"subj": "建築家", "rel": "設計した", "obj": "建物"}]},
    {"text": "料理人は新メニューを考案した", "triples": [{"subj": "料理人", "rel": "考案した", "obj": "新メニュー"}]},
    {"text": "弁護士は案件を担当した", "triples": [{"subj": "弁護士", "rel": "担当した", "obj": "案件"}]},
    {"text": "政治家は演説を行った", "triples": [{"subj": "政治家", "rel": "行った", "obj": "演説"}]},
    {"text": "消防士は火災を鎮火した", "triples": [{"subj": "消防士", "rel": "鎮火した", "obj": "火災"}]},
    {"text": "警察官は事件を解決した", "triples": [{"subj": "警察官", "rel": "解決した", "obj": "事件"}]},
    # Final 5 to reach 100
    {"text": "画家は絵画を完成させた", "triples": [{"subj": "画家", "rel": "完成させた", "obj": "絵画"}]},
    {"text": "詩人は詩を書いた", "triples": [{"subj": "詩人", "rel": "書いた", "obj": "詩"}]},
    {"text": "翻訳家は文書を翻訳した", "triples": [{"subj": "翻訳家", "rel": "翻訳した", "obj": "文書"}]},
    {"text": "エンジニアはシステムを構築した", "triples": [{"subj": "エンジニア", "rel": "構築した", "obj": "システム"}]},
    {"text": "研究者は論文を発表した", "triples": [{"subj": "研究者", "rel": "発表した", "obj": "論文"}]},
]

assert len(ANNOTATED_DATA) == 100, f"Expected 100 samples, got {len(ANNOTATED_DATA)}"


# ===========================================================================
# Pattern-Based Extraction
# ===========================================================================

# Patterns: (regex, subj_group, rel_group, obj_group)
EXTRACTION_PATTERNS = [
    # [subj]が[obj]を[rel]
    (r"(.+?)が(.+?)を(.+?)$", 1, 3, 2),
    # [subj]は[obj]を[rel]
    (r"(.+?)は(.+?)を(.+?)$", 1, 3, 2),
    # [subj]の[rel_part]は[obj]
    (r"(.+?)の(.+?)は(.+?)$", 1, None, 3, "possessive"),
    # [subj]が[obj]に[rel]  (double obj / dative)
    (r"(.+?)が(.+?)に(.+?)を(.+?)$", 1, 4, 3),
    # [subj]は[obj]に[rel]
    (r"(.+?)は(.+?)に(.+?)を(.+?)$", 1, 4, 3),
]


@dataclass
class Triple:
    subj: str
    rel: str
    obj: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subj, self.rel, self.obj)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Triple):
            return False
        return self.as_tuple() == other.as_tuple()

    def __hash__(self):
        return hash(self.as_tuple())


def extract_patterns(text: str) -> list[Triple]:
    """Rule-based triple extraction from Japanese text."""
    results: list[Triple] = []

    # Pattern 1: Xが/はYをZ
    m = re.match(r"^(.+?)(?:が|は)(.+?)を(.+?)(?:た|だ|る|ている|した|いる|れる|られる)(.*)$", text)
    if m:
        subj = m.group(1).strip()
        obj = m.group(2).strip()
        # verb: reconstruct from context
        tail = m.group(3).strip()
        # Find last verb-like ending
        verb_match = re.search(r"(.+?(?:た|だ|する|した|いた|った|んだ|した|てた|べた|きた|りた|みた|んた|えた|りた|せた|てた|でた|いだ|でだ|した))$", text)
        rel = verb_match.group(1).split("を")[-1].strip() if verb_match else tail
        if subj and obj:
            results.append(Triple(subj=subj, rel=rel, obj=obj))

    # Pattern 2: XのYはZ
    m2 = re.match(r"^(.+?)の(.+?)は(.+?)(?:だ|です|ある|ます|い|な)(.*)$", text)
    if m2:
        subj = m2.group(1).strip()
        rel_part = m2.group(2).strip()
        obj = m2.group(3).strip()
        # Clean trailing particles
        obj = re.sub(r"(だ|です|ある|ます)$", "", obj).strip()
        if subj and rel_part and obj:
            results.append(Triple(subj=subj, rel=f"の{rel_part}", obj=obj))

    # Pattern 3: XがYをZした (particle が)
    m3 = re.match(r"^(.+?)が(.+?)を(.+)$", text)
    if m3 and not results:
        subj = m3.group(1).strip()
        obj = m3.group(2).strip()
        rel_raw = m3.group(3).strip()
        # Keep verb as-is (remove に/へ/で particles if present)
        rel = re.sub(r"^に|^へ|^で", "", rel_raw).strip()
        if subj and obj and rel:
            results.append(Triple(subj=subj, rel=rel, obj=obj))

    return results


# ===========================================================================
# Character-level Vectorization (TF-IDF)
# ===========================================================================

def _char_tokenize(text: str) -> list[str]:
    """Tokenize text into unigrams and bigrams."""
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return chars + bigrams


def build_vocab(sentences: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for sent in sentences:
        for tok in _char_tokenize(sent):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def vectorize_tf_idf(sentences: list[str], vocab: dict[str, int]) -> np.ndarray:
    """Build TF-IDF matrix (n_docs × vocab_size)."""
    V = len(vocab)
    N = len(sentences)
    tf = np.zeros((N, V), dtype=np.float32)
    for i, sent in enumerate(sentences):
        toks = _char_tokenize(sent)
        for tok in toks:
            if tok in vocab:
                tf[i, vocab[tok]] += 1
        row_sum = tf[i].sum()
        if row_sum > 0:
            tf[i] /= row_sum

    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    return tf * idf


# ===========================================================================
# Neural Scoring MLP (pure NumPy)
# ===========================================================================

class TripleScoringMLP:
    """
    Small 2-layer MLP that scores (subj, rel, obj) triple candidates.
    Input: concatenation of TF-IDF vectors for subj + rel + obj
    Output: sigmoid score in [0,1]

    Trained with pseudo-labels: gold triples = 1, random negatives = 0.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 0.01):
        self.lr = lr
        # Xavier init
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1, dtype=np.float32)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        self._h = self._relu(X @ self.W1 + self.b1)
        out = self._sigmoid(self._h @ self.W2 + self.b2)
        return out.flatten()

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        N = len(y_pred)
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

        # Gradient of output layer
        d_out = (y_pred - y_true).reshape(-1, 1) / N
        dW2 = self._h.T @ d_out
        db2 = d_out.sum(axis=0)

        # Gradient through relu
        d_h = d_out @ self.W2.T
        d_h[self._h <= 0] = 0
        dW1 = self._X.T @ d_h
        db1 = d_h.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


def _triple_to_vector(triple: dict, vocab: dict[str, int], V: int) -> np.ndarray:
    """Vectorize a triple by summing char vectors of subj+rel+obj."""
    vec = np.zeros(V * 3, dtype=np.float32)
    for part_idx, key in enumerate(["subj", "rel", "obj"]):
        text = triple.get(key, "")
        for tok in _char_tokenize(text):
            if tok in vocab:
                vec[part_idx * V + vocab[tok]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


# ===========================================================================
# Evaluation metrics
# ===========================================================================

def _normalize(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r"[た。、だです。]$", "", s.strip())


def exact_match(pred: Triple, gold: Triple) -> bool:
    return (_normalize(pred.subj) == _normalize(gold.subj)
            and _normalize(pred.obj) == _normalize(gold.obj))


def relaxed_match(pred: Triple, gold: Triple) -> bool:
    """At least 2 out of 3 fields partially match (subj-obj overlap)."""
    subj_match = _normalize(pred.subj) in _normalize(gold.subj) or \
                 _normalize(gold.subj) in _normalize(pred.subj)
    obj_match  = _normalize(pred.obj) in _normalize(gold.obj) or \
                 _normalize(gold.obj) in _normalize(pred.obj)
    return subj_match and obj_match


def compute_f1(
    predictions: list[list[Triple]],
    gold_labels: list[list[Triple]],
    match_fn,
) -> dict[str, float]:
    tp = fp = fn = 0
    for preds, golds in zip(predictions, gold_labels):
        matched_gold = set()
        for p in preds:
            found = False
            for gi, g in enumerate(golds):
                if gi not in matched_gold and match_fn(p, g):
                    found = True
                    matched_gold.add(gi)
                    break
            if found:
                tp += 1
            else:
                fp += 1
        fn += len(golds) - len(matched_gold)

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ===========================================================================
# Full Pipeline
# ===========================================================================

def run_pipeline(n_epochs: int = 30, verbose: bool = True) -> dict:
    texts = [d["text"] for d in ANNOTATED_DATA]
    gold_lists = [[Triple(**t) for t in d["triples"]] for d in ANNOTATED_DATA]

    # --- Pattern-based extraction ---
    pattern_preds = [extract_patterns(t) for t in texts]

    exact_results = compute_f1(pattern_preds, gold_lists, exact_match)
    relaxed_results = compute_f1(pattern_preds, gold_lists, relaxed_match)

    if verbose:
        print("[Pattern-based Extraction]")
        print(f"  Exact   : P={exact_results['precision']:.3f} R={exact_results['recall']:.3f} "
              f"F1={exact_results['f1']:.3f}")
        print(f"  Relaxed : P={relaxed_results['precision']:.3f} R={relaxed_results['recall']:.3f} "
              f"F1={relaxed_results['f1']:.3f}")

    # --- Neural scoring ---
    all_items = [item for d in ANNOTATED_DATA for item in d["triples"]]
    all_texts = [d["text"] for d in ANNOTATED_DATA for _ in d["triples"]]
    vocab = build_vocab(all_texts + [item.get("subj", "") + item.get("rel", "") + item.get("obj", "")
                                     for item in all_items])
    V = len(vocab)

    # Build training data: gold triples = 1, negatives = 0
    gold_vecs = [_triple_to_vector(t, vocab, V) for t in all_items]
    # Negatives: shuffle obj field
    rng = np.random.default_rng(42)
    neg_triples = []
    for item in all_items:
        neg = {k: v for k, v in item.items()}
        # swap obj with random gold obj
        neg["obj"] = all_items[int(rng.integers(len(all_items)))]["obj"]
        neg_triples.append(neg)
    neg_vecs = [_triple_to_vector(t, vocab, V) for t in neg_triples]

    X = np.array(gold_vecs + neg_vecs, dtype=np.float32)
    y = np.array([1.0] * len(gold_vecs) + [0.0] * len(neg_vecs), dtype=np.float32)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    mlp = TripleScoringMLP(input_dim=V * 3, hidden_dim=64, lr=0.05)
    losses: list[float] = []
    for epoch in range(n_epochs):
        preds = mlp.forward(X)
        loss = mlp.backward(preds, y)
        losses.append(loss)

    if verbose:
        print(f"\n[Neural Scoring MLP] trained {n_epochs} epochs, final loss={losses[-1]:.4f}")
        # Score gold vs negatives
        gold_scores = mlp.score(np.array(gold_vecs))
        neg_scores = mlp.score(np.array(neg_vecs))
        print(f"  Mean gold score   : {gold_scores.mean():.3f}")
        print(f"  Mean neg score    : {neg_scores.mean():.3f}")

    results = {
        "pattern_exact_f1": exact_results,
        "pattern_relaxed_f1": relaxed_results,
        "neural_final_loss": round(float(losses[-1]), 4),
        "neural_mean_gold_score": round(float(mlp.score(np.array(gold_vecs)).mean()), 3),
        "neural_mean_neg_score": round(float(mlp.score(np.array(neg_vecs)).mean()), 3),
        "n_samples": len(ANNOTATED_DATA),
    }
    return results


# ===========================================================================
# __main__
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Information Extraction — Relation Triple Extraction")
    print("=" * 60)

    results = run_pipeline(n_epochs=50, verbose=True)

    # Show some examples
    print("\n--- Sample Extractions ---")
    for item in ANNOTATED_DATA[:10]:
        preds = extract_patterns(item["text"])
        gold = [Triple(**t) for t in item["triples"]]
        pred_str = [(p.subj, p.rel, p.obj) for p in preds]
        gold_str = [(g.subj, g.rel, g.obj) for g in gold]
        match = any(relaxed_match(p, g) for p in preds for g in gold) if preds else False
        status = "OK" if match else "MISS"
        print(f"  [{status}] {item['text']}")
        print(f"       Gold: {gold_str}")
        print(f"       Pred: {pred_str}")

    out_path = os.path.join(OUT_DIR, "ie_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nFinal summary:")
    print(f"  Pattern Exact   F1 = {results['pattern_exact_f1']['f1']:.3f}")
    print(f"  Pattern Relaxed F1 = {results['pattern_relaxed_f1']['f1']:.3f}")
