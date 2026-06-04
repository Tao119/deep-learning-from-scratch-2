"""
nlp_applications/machine_translation.py — Japanese-English Machine Translation

Full seq2seq translation system with attention.

Dataset  : 300 synthetic Japanese-English sentence pairs (medical / daily / technical)
Tokenizer: BPE (from common/bpe.py)
Model    : AttentionSeq2Seq (from ch08/attention.py)
Training : 200 epochs, BLEU score tracked
Results  : saved to experiments/04-translation/

Usage
-----
    cd nlp_applications
    python machine_translation.py
"""

import sys
import os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import math
from collections import Counter

import numpy as np

from common.bpe import BPETokenizer
from common.optimizer import Adam
from common.util import clip_grads

# AttentionSeq2Seq lives in ch08/attention.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch08"))
from attention import AttentionSeq2Seq


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

MEDICAL_TEMPLATES = [
    ("患者は{症状}を訴えている", "The patient complains of {symptom}"),
    ("患者は{症状}がある", "The patient has {symptom}"),
    ("{症状}の患者が来院した", "A patient with {symptom} presented"),
    ("この患者に{治療}を行う", "We perform {treatment} on this patient"),
    ("血圧は{数値}mmHgである", "The blood pressure is {value} mmHg"),
    ("体温は{数値}度である", "The temperature is {value} degrees"),
    ("脈拍は{数値}回/分である", "The pulse is {value} beats per minute"),
    ("この薬は{症状}に有効である", "This medication is effective for {symptom}"),
    ("手術は{時間}時間かかる", "The surgery takes {time} hours"),
    ("検査結果は{結果}を示している", "The test results indicate {result}"),
]

MEDICAL_FILLERS = {
    "症状": ["頭痛", "発熱", "胸痛", "腹痛", "倦怠感", "咳", "息切れ", "嘔気"],
    "symptom": ["headache", "fever", "chest pain", "abdominal pain",
                "fatigue", "cough", "shortness of breath", "nausea"],
    "治療": ["手術", "化学療法", "放射線治療", "リハビリ", "投薬"],
    "treatment": ["surgery", "chemotherapy", "radiation therapy",
                  "rehabilitation", "medication"],
    "数値": ["80", "90", "100", "110", "120", "130", "36", "37", "38", "39"],
    "value": ["80", "90", "100", "110", "120", "130", "36", "37", "38", "39"],
    "時間": ["一", "二", "三", "四", "五"],
    "time": ["one", "two", "three", "four", "five"],
    "結果": ["正常", "異常", "陽性", "陰性"],
    "result": ["normal", "abnormal", "positive", "negative"],
}

DAILY_TEMPLATES = [
    ("今日の天気は{天気}です", "Today's weather is {weather}"),
    ("明日の天気は{天気}でしょう", "Tomorrow's weather will be {weather}"),
    ("今朝は{食事}を食べた", "I ate {meal} this morning"),
    ("今夜は{食事}を食べる予定だ", "I plan to eat {meal} tonight"),
    ("電車が{時間}分遅れている", "The train is {time} minutes late"),
    ("今日は{天気}なので外出を控えた", "Today is {weather} so I stayed home"),
    ("昨日の会議は{感想}だった", "Yesterday's meeting was {impression}"),
    ("この映画は{感想}だった", "This movie was {impression}"),
    ("明日は{活動}をする予定だ", "I plan to {activity} tomorrow"),
    ("週末に{活動}をした", "I {activity} on the weekend"),
]

DAILY_FILLERS = {
    "天気": ["晴れ", "曇り", "雨", "雪", "強風", "霧"],
    "weather": ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"],
    "食事": ["ご飯", "パン", "麺", "サラダ", "スープ"],
    "meal": ["rice", "bread", "noodles", "salad", "soup"],
    "時間": ["五", "十", "十五", "二十", "三十"],
    "time": ["five", "ten", "fifteen", "twenty", "thirty"],
    "感想": ["面白い", "退屈", "有益", "無駄", "感動的"],
    "impression": ["interesting", "boring", "productive", "useless", "moving"],
    "活動": ["旅行", "ハイキング", "読書", "料理", "掃除"],
    "activity": ["travel", "hike", "read", "cook", "clean"],
}

TECHNICAL_TEMPLATES = [
    ("このシステムは{機能}を提供する", "This system provides {feature}"),
    ("このソフトウェアは{機能}に対応している", "This software supports {feature}"),
    ("{機能}を実装するために{技術}を使用した", "We used {technology} to implement {feature}"),
    ("このAPIは{機能}を可能にする", "This API enables {feature}"),
    ("データベースに{データ}を保存する", "Store {data} in the database"),
    ("サーバーは{状態}である", "The server is {status}"),
    ("{技術}を使用してシステムを構築した", "We built the system using {technology}"),
    ("このアルゴリズムは{処理}を最適化する", "This algorithm optimizes {process}"),
    ("エラーログに{エラー}が記録された", "The error log recorded {error}"),
    ("システムの応答時間は{時間}ミリ秒である", "The system response time is {time} milliseconds"),
]

TECHNICAL_FILLERS = {
    "機能": ["認証", "暗号化", "データ処理", "ログ管理", "負荷分散"],
    "feature": ["authentication", "encryption", "data processing",
                "log management", "load balancing"],
    "技術": ["Python", "Java", "機械学習", "クラウド", "API"],
    "technology": ["Python", "Java", "machine learning", "cloud computing", "APIs"],
    "データ": ["ユーザー情報", "トランザクション", "ログデータ", "設定ファイル"],
    "data": ["user information", "transactions", "log data", "configuration files"],
    "状態": ["正常稼働中", "メンテナンス中", "停止中", "再起動中"],
    "status": ["running normally", "under maintenance", "stopped", "restarting"],
    "処理": ["クエリ処理", "データ変換", "メモリ使用", "スループット"],
    "process": ["query processing", "data transformation", "memory usage", "throughput"],
    "エラー": ["接続タイムアウト", "認証失敗", "メモリ不足", "ディスク容量不足"],
    "error": ["connection timeout", "authentication failure",
              "out of memory", "insufficient disk space"],
    "時間": ["50", "100", "200", "500", "1000"],
    "time": ["50", "100", "200", "500", "1000"],
}


def _expand_template(template_ja, template_en, fillers_ja, fillers_en, n=10, seed=0):
    """
    Enumerate up to `n` combinations of the template slots.
    Returns list of (ja, en) pairs.
    """
    rng = np.random.default_rng(seed)
    # Find slot names in both templates
    slots_ja = re.findall(r"\{([^}]+)\}", template_ja)
    slots_en = re.findall(r"\{([^}]+)\}", template_en)

    pairs = []
    for _ in range(n):
        ja = template_ja
        en = template_en
        for s_ja, s_en in zip(slots_ja, slots_en):
            choices_ja = fillers_ja.get(s_ja, [s_ja])
            choices_en = fillers_en.get(s_en, [s_en])
            idx = rng.integers(len(choices_ja))
            idx_en = min(idx, len(choices_en) - 1)
            ja = ja.replace("{" + s_ja + "}", choices_ja[idx], 1)
            en = en.replace("{" + s_en + "}", choices_en[idx_en], 1)
        pairs.append((ja, en))
    return pairs


def build_corpus(n_per_template=10):
    """
    Build ~300 Japanese-English sentence pairs.
    """
    pairs = []

    for t_ja, t_en in MEDICAL_TEMPLATES:
        pairs.extend(_expand_template(
            t_ja, t_en, MEDICAL_FILLERS, MEDICAL_FILLERS, n=n_per_template, seed=1
        ))

    for t_ja, t_en in DAILY_TEMPLATES:
        pairs.extend(_expand_template(
            t_ja, t_en, DAILY_FILLERS, DAILY_FILLERS, n=n_per_template, seed=2
        ))

    for t_ja, t_en in TECHNICAL_TEMPLATES:
        pairs.extend(_expand_template(
            t_ja, t_en, TECHNICAL_FILLERS, TECHNICAL_FILLERS, n=n_per_template, seed=3
        ))

    return pairs


# ---------------------------------------------------------------------------
# BPE tokenizer setup (character-level BPE on this tiny corpus)
# ---------------------------------------------------------------------------

def build_tokenizer(corpus_texts, vocab_size=300):
    tok = BPETokenizer()
    tok.train(corpus_texts, vocab_size=vocab_size, verbose=False)
    return tok


def add_special_tokens(tok, pad_id=0, sos_id=1, eos_id=2):
    """
    Inject <pad>, <sos>, <eos> at fixed positions.
    Returns updated (tok, pad_id, sos_id, eos_id).
    """
    specials = ["<pad>", "<sos>", "<eos>"]
    for i, sp in enumerate(specials):
        if sp not in tok.vocab:
            tok.vocab[sp] = i
            tok.inv_vocab[i] = sp
    return tok, 0, 1, 2


# ---------------------------------------------------------------------------
# Encode / pad helpers
# ---------------------------------------------------------------------------

def encode_pair(tok, ja, en, sos_id, eos_id, max_len=40):
    """Encode one (ja, en) pair into padded integer arrays."""
    src = tok.encode(ja)[:max_len]
    tgt = tok.encode(en)[:max_len]
    return src, tgt


def pad_seqs(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    padded = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded


# ---------------------------------------------------------------------------
# BLEU score (corpus-level)
# ---------------------------------------------------------------------------

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hypothesis, reference, max_n=4):
    """Compute sentence-level BLEU (smoothed)."""
    if len(hypothesis) == 0:
        return 0.0

    score = 1.0
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        ref_counts = _ngram_counts(reference, n)
        if not hyp_counts:
            score *= 0.0
            continue
        clipped = sum(min(c, ref_counts.get(ng, 0)) for ng, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        # Add-1 smoothing to avoid zero precision
        precision = (clipped + 1) / (total + 1)
        score *= precision

    # Brevity penalty
    bp = 1.0 if len(hypothesis) >= len(reference) else math.exp(
        1 - len(reference) / max(len(hypothesis), 1)
    )
    return bp * (score ** (1 / max_n))


def corpus_bleu(hypotheses, references, max_n=4):
    """Average sentence BLEU over a list of (hyp, ref) pairs."""
    scores = [sentence_bleu(h, r, max_n) for h, r in zip(hypotheses, references)]
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_batches(src_seqs, tgt_seqs, sos_id, eos_id, batch_size=32):
    """
    Yield (src_batch, tgt_batch) pairs.
    tgt_batch shape: (B, T+1) with <sos> prepended.
    """
    n = len(src_seqs)
    idx = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        src_list = [src_seqs[i] for i in batch_idx]
        tgt_list = [[sos_id] + list(tgt_seqs[i]) + [eos_id] for i in batch_idx]
        yield pad_seqs(src_list), pad_seqs(tgt_list)


def translate(model, src, tok, sos_id, eos_id, max_len=40):
    """Greedy decoding for one source sequence."""
    src_batch = np.array(src, dtype=np.int32)[np.newaxis]
    gen = model.generate(src_batch, start_id=sos_id, sample_size=max_len)
    # Truncate at first EOS
    tokens = []
    for t in gen:
        if t == eos_id:
            break
        tokens.append(t)
    return tok.decode(tokens)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results(out_dir, history, test_results):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump({"loss_history": history["loss"],
                   "bleu_history": history["bleu"]}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test_translations.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # ---- Build corpus ----
    print("Building corpus…")
    pairs = build_corpus(n_per_template=10)
    print(f"  Total pairs: {len(pairs)}")

    # Shuffle and split
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    n_test = 30
    train_pairs = pairs[n_test:]
    test_pairs = pairs[:n_test]

    all_texts = [ja for ja, _ in pairs] + [en for _, en in pairs]

    # ---- BPE tokenizer ----
    print("Training BPE tokenizer…")
    tok = build_tokenizer(all_texts, vocab_size=200)
    tok, PAD, SOS, EOS = add_special_tokens(tok)
    V = len(tok.vocab)
    print(f"  Vocab size: {V}")

    # ---- Encode corpus ----
    train_src = [tok.encode(ja) for ja, _ in train_pairs]
    train_tgt = [tok.encode(en) for _, en in train_pairs]
    test_src = [tok.encode(ja) for ja, _ in test_pairs]
    test_tgt = [tok.encode(en) for _, en in test_pairs]

    # ---- Model ----
    wordvec_size = 64
    hidden_size = 128
    model = AttentionSeq2Seq(V, wordvec_size, hidden_size)
    optimizer = Adam(lr=5e-4)

    # ---- Training ----
    max_epoch = 200
    batch_size = 32
    history = {"loss": [], "bleu": []}

    print(f"\nTraining AttentionSeq2Seq for {max_epoch} epochs…")
    for epoch in range(max_epoch):
        total_loss = 0
        n_batches = 0
        for src_b, tgt_b in make_batches(train_src, train_tgt,
                                          SOS, EOS, batch_size=batch_size):
            model.encoder_lstm.reset_state()
            model.decoder_lstm.reset_state()
            loss = model.forward(src_b, tgt_b)
            model.backward()
            clip_grads(model.grads, 5.0)
            optimizer.update(model.params, model.grads)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["loss"].append(float(avg_loss))

        # BLEU on a small validation sample every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == max_epoch - 1:
            hyps = []
            refs = []
            for src, tgt in zip(test_src[:10], test_tgt[:10]):
                model.encoder_lstm.reset_state()
                hyp_text = translate(model, src, tok, SOS, EOS)
                hyp_ids = tok.encode(hyp_text)
                hyps.append(hyp_ids)
                refs.append(tgt)
            bleu = corpus_bleu(hyps, refs)
            history["bleu"].append(round(bleu, 4))
            print(f"  epoch {epoch + 1:>3}  loss={avg_loss:.4f}  BLEU={bleu:.4f}")
        else:
            history["bleu"].append(history["bleu"][-1] if history["bleu"] else 0.0)

    # ---- Demo: translate 10 test sentences ----
    print("\n=== Translation Demo (10 sentences) ===")
    test_results = []
    for i, ((ja, en_ref), src, tgt) in enumerate(
            zip(test_pairs[:10], test_src[:10], test_tgt[:10])):
        model.encoder_lstm.reset_state()
        en_hyp = translate(model, src, tok, SOS, EOS)
        hyp_ids = tok.encode(en_hyp)
        bleu = sentence_bleu(hyp_ids, tgt)
        print(f"[{i + 1:>2}]  JP : {ja}")
        print(f"      REF: {en_ref}")
        print(f"      HYP: {en_hyp}  (BLEU={bleu:.3f})")
        print()
        test_results.append({
            "japanese": ja,
            "reference": en_ref,
            "hypothesis": en_hyp,
            "bleu": round(bleu, 4),
        })

    # ---- Save ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments", "04-translation")
    save_results(out_dir, history, test_results)

    final_bleu = corpus_bleu(
        [tok.encode(translate(model, s, tok, SOS, EOS)) for s in test_src[:10]],
        test_tgt[:10]
    )
    print(f"Final corpus BLEU on 10 test sentences: {final_bleu:.4f}")
    print("Done.")
