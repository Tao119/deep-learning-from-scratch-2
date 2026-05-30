"""
Beam Search decoder on top of TransformerLM.

Compares greedy vs beam search generation on the same small corpus used
in transformer_lm.py, showing 5 generated sequences with log-probability
scores for each strategy.
"""

import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, clip_grads
from common.optimizer import Adam
from ch08.transformer_lm import TransformerLM, run_training


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Greedy decoder
# ---------------------------------------------------------------------------

def greedy_search(model, start_id, max_len=20):
    """
    Greedy autoregressive decoding.

    Returns
    -------
    tokens   : list of int, length max_len
    log_prob : float, sum of log-probs of chosen tokens
    """
    tokens   = [start_id]
    log_prob = 0.0

    for _ in range(max_len - 1):
        xs = np.array(tokens, dtype=np.int32)[np.newaxis]   # (1, t)
        T  = xs.shape[1]
        x  = model.embed_W[xs] + model.pe[:T]
        for blk in model.blocks:
            x = blk.forward(x)
        logits = x[0, -1] @ model.embed_W.T + model.head_b  # (V,)
        probs  = _softmax(logits)
        next_id = int(logits.argmax())
        log_prob += float(np.log(probs[next_id] + 1e-12))
        tokens.append(next_id)

    return tokens, log_prob


# ---------------------------------------------------------------------------
# Beam Search decoder
# ---------------------------------------------------------------------------

def beam_search(model, start_id, beam_width=3, max_len=20,
                word_to_id=None, id_to_word=None):
    """
    Beam search autoregressive decoding over a TransformerLM.

    Parameters
    ----------
    model      : TransformerLM instance (already trained)
    start_id   : int, the BOS / seed token id
    beam_width : int, number of hypotheses kept at each step
    max_len    : int, total sequence length (including start token)
    word_to_id : dict (optional, used only for debug prints)
    id_to_word : dict (optional, used only for debug prints)

    Returns
    -------
    best_tokens   : list of int, the highest-scoring hypothesis
    all_beams     : list of (tokens, log_prob) for every finished beam,
                    sorted best-first
    """
    # Each beam is (log_prob, token_list)
    beams = [(0.0, [start_id])]

    for step in range(max_len - 1):
        candidates = []

        for log_prob, tokens in beams:
            xs = np.array(tokens, dtype=np.int32)[np.newaxis]  # (1, t)
            T  = xs.shape[1]
            x  = model.embed_W[xs] + model.pe[:T]
            for blk in model.blocks:
                x = blk.forward(x)
            logits = x[0, -1] @ model.embed_W.T + model.head_b  # (V,)
            log_probs_step = np.log(_softmax(logits) + 1e-12)   # (V,)

            # Take the top beam_width next tokens
            top_ids = np.argsort(log_probs_step)[-beam_width:][::-1]
            for nid in top_ids:
                new_log_prob = log_prob + float(log_probs_step[nid])
                candidates.append((new_log_prob, tokens + [int(nid)]))

        # Prune to beam_width best candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Sort finished beams best-first
    beams.sort(key=lambda x: x[0], reverse=True)
    best_log_prob, best_tokens = beams[0]
    return best_tokens, beams


# ---------------------------------------------------------------------------
# Main: train + compare greedy vs beam search
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text = ("the dog ran . the cat sat . the dog sat . "
            "a cat ran . a dog ran . the cat ran . "
            "the dog ate . a cat ate . the cat ate .")
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)

    # Tiny Transformer — same hyper-params as transformer_lm.py
    d_model    = 32
    n_heads    = 4
    n_layers   = 2
    d_ff       = 64
    batch_size = 4
    time_size  = 5
    max_epoch  = 200

    print("=" * 60)
    print("Training TransformerLM …")
    print("=" * 60)
    model, _ = run_training(
        corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
        batch_size, time_size, max_epoch, label="Transformer"
    )

    # Seed tokens to generate from
    seed_words = ["the", "a", "dog", "cat", "the"]
    gen_len    = 8
    beam_width = 3

    print("\n" + "=" * 60)
    print(f"Generation comparison (beam_width={beam_width}, length={gen_len})")
    print("=" * 60)

    for seed in seed_words:
        start_id = word_to_id.get(seed, 0)

        # --- Greedy ---
        g_tokens, g_lp = greedy_search(model, start_id, max_len=gen_len)
        g_words = [id_to_word.get(t, "?") for t in g_tokens]

        # --- Beam search ---
        b_tokens, all_beams = beam_search(
            model, start_id,
            beam_width=beam_width,
            max_len=gen_len,
            word_to_id=word_to_id,
            id_to_word=id_to_word,
        )
        b_words = [id_to_word.get(t, "?") for t in b_tokens]
        b_lp    = all_beams[0][0]

        print(f"\nSeed: '{seed}'")
        print(f"  Greedy : {' '.join(g_words):<45}  log_p={g_lp:.3f}")
        print(f"  Beam   : {' '.join(b_words):<45}  log_p={b_lp:.3f}")

    # -----------------------------------------------------------------------
    # Show 5 generated sequences with log-prob scores (beam search)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Top-5 beam search sequences from seed 'the'")
    print("=" * 60)
    start_id    = word_to_id.get("the", 0)
    _, all_beams = beam_search(
        model, start_id,
        beam_width=5,
        max_len=gen_len,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
    )
    for rank, (lp, tokens) in enumerate(all_beams[:5], 1):
        words = [id_to_word.get(t, "?") for t in tokens]
        print(f"  #{rank}  log_p={lp:>7.3f}  |  {' '.join(words)}")
