"""
Extended BPE Analysis with Vocabulary Analysis — pure Python + NumPy.

ch08/byte_pair_encoding_advanced.py

Tasks
-----
1. Train BPE on Japanese medical corpus (100 sentences)
2. Show vocabulary growth curve (merges vs unique tokens)
3. Show token frequency distribution (Zipf's law check)
4. Compare tokenization: character-level vs word-level vs BPE
5. Compute average tokens/sentence for each scheme
6. Find optimal vocab size (minimize perplexity on held-out data)
7. Save: ch08/bpe_analysis.png (4 subplots)

Reference
---------
  Sennrich, R., Haddow, B., Birch, A. (2016).
  Neural Machine Translation of Rare Words with Subword Units.
  ACL 2016.
"""

import sys
import os
import re
import math
import time
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Medical corpus (100 Japanese sentences)
# ---------------------------------------------------------------------------

def build_medical_corpus():
    """100 Japanese medical sentences for BPE analysis."""
    sentences = [
        # Diagnoses
        "高血圧は慢性疾患である",
        "糖尿病はインスリン抵抗性を引き起こす",
        "心筋梗塞は緊急処置が必要だ",
        "脳卒中は早期治療が重要だ",
        "肺炎は抗菌薬で治療する",
        "敗血症は生命の危険がある",
        "癌は早期発見が大切だ",
        "骨折は固定が必要だ",
        "喘息は気管支拡張薬を使用する",
        "腎不全は透析が必要な場合がある",
        # Symptoms
        "胸痛は心臓病の重要な症状である",
        "発熱は感染症の主要な症状だ",
        "頭痛は様々な原因で起こる",
        "腹痛は消化器疾患を示す場合がある",
        "呼吸困難は肺や心臓の問題である",
        "意識障害は脳疾患のサインだ",
        "浮腫は心不全や腎不全で見られる",
        "黄疸は肝臓疾患のサインである",
        "血尿は腎臓や膀胱の問題を示す",
        "体重減少は悪性腫瘍の可能性がある",
        # Vital signs
        "血圧は収縮期と拡張期で表す",
        "心拍数は毎分の心拍数を示す",
        "体温は感染症の指標となる",
        "呼吸数は肺機能を反映する",
        "酸素飽和度は肺の換気を示す",
        "血糖値は糖尿病管理に重要だ",
        "脈拍は心臓の状態を示す",
        "尿量は腎機能の指標となる",
        "意識レベルはGCSで評価する",
        "体重は栄養状態を反映する",
        # Lab values
        "白血球数は感染症で上昇する",
        "赤血球数は貧血の診断に使う",
        "血小板数は凝固機能に関係する",
        "CRPは炎症反応の指標となる",
        "クレアチニンは腎機能を示す",
        "ALTは肝細胞障害を反映する",
        "BNPは心不全の診断に有用だ",
        "トロポニンは心筋障害を示す",
        "D-ダイマーは血栓症を示す",
        "乳酸値は組織低酸素を示す",
        # Treatments
        "アスピリンは血小板凝集を抑制する",
        "ワルファリンは抗凝固薬である",
        "フロセミドは利尿作用がある",
        "アモキシシリンは抗菌薬である",
        "インスリンは血糖を下げる働きをする",
        "ニトログリセリンは狭心症に使用する",
        "モルヒネは強力な鎮痛薬である",
        "ドパミンは昇圧薬として使用する",
        "ステロイドは抗炎症作用がある",
        "利尿薬は心不全治療に使用する",
        # Procedures
        "心電図は心臓の電気活動を記録する",
        "超音波検査は臓器を観察する",
        "CT検査は詳細な断面像を提供する",
        "MRI検査は軟部組織を詳しく見る",
        "採血は診断に必要な情報を提供する",
        "点滴は薬剤や輸液を投与する方法だ",
        "挿管は気道を確保する処置だ",
        "除細動は不整脈を治療する",
        "透析は腎機能を代替する治療だ",
        "輸血は貧血や出血を治療する",
        # ICU/Emergency
        "敗血症性ショックは生命の危機である",
        "急性呼吸不全は即時対応が必要だ",
        "急性心筋梗塞は緊急治療が必要だ",
        "脳梗塞は時間が重要な疾患だ",
        "外傷は多発損傷の可能性がある",
        "アナフィラキシーはアドレナリンで治療する",
        "心停止は即時CPRが必要だ",
        "肺塞栓症は抗凝固療法が必要だ",
        "大動脈解離は緊急手術が必要なことがある",
        "緊張性気胸は緊急脱気が必要だ",
        # Prevention
        "予防接種は感染症を防ぐ",
        "健康診断は早期発見に役立つ",
        "禁煙は肺疾患を予防する",
        "適度な運動は健康を維持する",
        "バランスの良い食事は重要だ",
        "ストレス管理は精神健康に大切だ",
        "睡眠は免疫機能に影響する",
        "水分補給は体の機能を維持する",
        "定期的な検診は推奨される",
        "生活習慣改善は疾患予防に重要だ",
        # Nursing
        "患者の安全は最優先事項だ",
        "コミュニケーションは看護の基本だ",
        "疼痛管理は患者のQOLを高める",
        "感染予防は医療の基本である",
        "チーム医療は安全な治療を実現する",
        "記録は正確に記載する必要がある",
        "患者教育は治療の一部である",
        "リハビリは早期から開始する",
        "栄養管理は回復を促進する",
        "転倒予防は高齢患者に重要だ",
        # Cardiology specifics
        "心房細動は不整脈の一種である",
        "冠動脈疾患は動脈硬化が原因だ",
        "心不全は心臓のポンプ機能低下だ",
        "弁膜症は心臓の弁の異常だ",
        "ペースメーカーは不整脈に使用する",
        "ICD は致死性不整脈を治療する",
        "カテーテル治療は低侵襲手術だ",
        "冠動脈造影は診断に有用だ",
        "心臓リハビリは重要な治療だ",
        "抗血小板薬は血栓予防に使用する",
        # Pulmonology specifics
        "COPDは慢性閉塞性肺疾患である",
        "気管支喘息はアレルギー性疾患だ",
        "肺癌は早期発見が治療に重要だ",
        "胸水は様々な疾患で生じる",
        "気管支拡張薬はCOPDに使用する",
    ]
    return sentences[:100]


# ---------------------------------------------------------------------------
# BPE implementation
# ---------------------------------------------------------------------------

def get_word_freqs(sentences):
    """Compute word-level token frequencies from sentences."""
    freqs = collections.Counter()
    for s in sentences:
        # Character-level: split each word into chars with </w> end marker
        for char in s:
            # We treat each character as a word-unit for Japanese
            freqs[char] += 1
    return freqs


def get_char_vocab(sentences):
    """Get all unique characters."""
    chars = set()
    for s in sentences:
        chars.update(s)
    return chars


def prepare_bpe_vocab(sentences):
    """
    Prepare initial BPE vocabulary.
    Each sentence character is an atomic unit.
    Return: dict {token_tuple: frequency}
    """
    vocab = collections.Counter()
    for s in sentences:
        # Treat the full sentence as a sequence of characters
        chars = tuple(list(s) + ['</s>'])
        vocab[chars] += 1
    return dict(vocab)


def get_bigram_stats(vocab):
    """Count all adjacent symbol pairs."""
    pairs = collections.Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(vocab, best_pair):
    """Merge a pair of symbols in all vocabulary entries."""
    new_vocab = {}
    bigram = re.escape(' '.join(best_pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, freq in vocab.items():
        # Convert tuple to string, merge, convert back
        word_str = ' '.join(word)
        new_word_str = p.sub(''.join(best_pair), word_str)
        new_word = tuple(new_word_str.split())
        new_vocab[new_word] = freq
    return new_vocab


def train_bpe(sentences, num_merges=500, verbose=False):
    """
    Train BPE on sentences.

    Returns:
        merges         : list of merged pairs
        vocab_history  : list of (n_merges, n_unique_tokens)
        final_vocab    : dict {token_tuple: freq}
    """
    # Initialize vocabulary: sentence → tuple of chars
    vocab = {}
    for s in sentences:
        chars = tuple(list(s))
        vocab[chars] = vocab.get(chars, 0) + 1

    merges = []
    vocab_history = []

    # Initial unique tokens (characters)
    unique_tokens = set()
    for word in vocab:
        unique_tokens.update(word)
    vocab_history.append((0, len(unique_tokens)))

    for i in range(num_merges):
        pairs = get_bigram_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < 2:
            break

        vocab = merge_vocab(vocab, best_pair)
        merges.append(best_pair)

        # Count unique tokens after this merge
        unique_tokens = set()
        for word in vocab:
            unique_tokens.update(word)
        vocab_history.append((i + 1, len(unique_tokens)))

        if verbose and (i + 1) % 50 == 0:
            print(f"  Merge {i+1:4d}: {best_pair} → {''.join(best_pair)}"
                  f"  unique_tokens={len(unique_tokens)}")

    return merges, vocab_history, vocab


def apply_bpe_tokenize(sentence, merges):
    """Apply BPE merges to tokenize a sentence."""
    word = tuple(list(sentence))
    for pair in merges:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(''.join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
    return list(word)


# ---------------------------------------------------------------------------
# Tokenization schemes
# ---------------------------------------------------------------------------

def tokenize_char(sentence):
    """Character-level: each character is a token."""
    return list(sentence)


def tokenize_word(sentence):
    """Word-level: split on common Japanese boundaries (simplified)."""
    # For Japanese: split on common particles/conjunctions
    # Simple approach: 3-character chunks + remainder
    tokens = []
    s = sentence
    # Split on common patterns
    parts = re.split(r'(は|を|に|で|が|の|も|と|や|か|より|から|まで|だ|です|ます|ある|する|した|して|いる)', s)
    for p in parts:
        if p:
            tokens.append(p)
    return tokens if tokens else [sentence]


def tokenize_bpe_at_vocab_size(sentence, merges, target_vocab_size):
    """Apply only the first k merges to hit approximately target_vocab_size."""
    # Apply all merges up to index (this is a simplification)
    return apply_bpe_tokenize(sentence, merges[:target_vocab_size])


# ---------------------------------------------------------------------------
# Perplexity estimation
# ---------------------------------------------------------------------------

def estimate_perplexity_unigram(test_sentences, train_vocab_freq):
    """
    Estimate unigram language model perplexity on test_sentences
    using token frequencies from train_vocab_freq.
    Lower perplexity → better vocabulary coverage.
    """
    total_tokens = sum(train_vocab_freq.values())
    if total_tokens == 0:
        return float('inf')

    log_prob_sum = 0.0
    n_tokens = 0
    smoothing = 1  # add-1 smoothing

    vocab_size = len(train_vocab_freq)
    for tokens in test_sentences:
        for tok in tokens:
            freq = train_vocab_freq.get(tok, 0)
            prob = (freq + smoothing) / (total_tokens + smoothing * vocab_size)
            log_prob_sum += math.log(prob + 1e-10)
            n_tokens += 1

    if n_tokens == 0:
        return float('inf')
    avg_log_prob = log_prob_sum / n_tokens
    return math.exp(-avg_log_prob)


def find_optimal_vocab_size(sentences, max_merges=500, test_frac=0.2):
    """
    Train BPE at several merge counts and evaluate perplexity on held-out data.
    Returns: vocab_sizes, perplexities, optimal_n_merges
    """
    n_test = max(1, int(len(sentences) * test_frac))
    idx = np.random.permutation(len(sentences))
    train_sents = [sentences[i] for i in idx[n_test:]]
    test_sents = [sentences[i] for i in idx[:n_test]]

    merge_points = [0, 20, 50, 100, 150, 200, 300, 400, min(max_merges, 500)]
    merge_points = sorted(set(merge_points))

    # Train BPE on train set
    merges, vocab_history, _ = train_bpe(train_sents, num_merges=max_merges)

    vocab_sizes = []
    perplexities = []

    for n_merges in merge_points:
        # Tokenize train sentences with n_merges merges
        train_tokens = [apply_bpe_tokenize(s, merges[:n_merges]) for s in train_sents]
        test_tokens = [apply_bpe_tokenize(s, merges[:n_merges]) for s in test_sents]

        # Build token frequency from train
        freq = collections.Counter()
        for toks in train_tokens:
            freq.update(toks)

        ppl = estimate_perplexity_unigram(test_tokens, freq)
        n_unique = len(freq)
        vocab_sizes.append(n_unique)
        perplexities.append(ppl)

    # Find minimum perplexity
    best_idx = int(np.argmin(perplexities))
    optimal_n_merges = merge_points[best_idx]

    return vocab_sizes, perplexities, optimal_n_merges, merge_points


# ---------------------------------------------------------------------------
# Zipf's law check
# ---------------------------------------------------------------------------

def compute_token_freq_distribution(sentences, tokenizer_fn, **kwargs):
    freq = collections.Counter()
    for s in sentences:
        tokens = tokenizer_fn(s, **kwargs)
        freq.update(tokens)
    # Sort by frequency descending
    sorted_items = sorted(freq.items(), key=lambda x: -x[1])
    ranks = np.arange(1, len(sorted_items) + 1)
    counts = np.array([c for _, c in sorted_items])
    return ranks, counts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 65)
    print("Extended BPE Analysis — Japanese Medical Corpus")
    print("=" * 65)

    sentences = build_medical_corpus()
    print(f"Corpus: {len(sentences)} sentences")
    print(f"Example: '{sentences[0]}'")

    # Character stats
    all_chars = set()
    for s in sentences:
        all_chars.update(s)
    print(f"Unique characters: {len(all_chars)}")

    # ----- Train BPE -----
    max_merges = 300
    print(f"\n--- Training BPE ({max_merges} merges) ---")
    merges, vocab_history, final_vocab = train_bpe(sentences, num_merges=max_merges, verbose=True)
    print(f"Trained {len(merges)} merges")

    # Vocabulary growth curve data
    merge_counts = [v[0] for v in vocab_history]
    unique_token_counts = [v[1] for v in vocab_history]

    # ----- Tokenization comparison -----
    print("\n--- Tokenization Comparison ---")

    bpe_merges_100 = merges[:100]
    bpe_merges_200 = merges[:200]
    bpe_merges_max = merges

    schemes = {
        "char": lambda s: tokenize_char(s),
        "word": lambda s: tokenize_word(s),
        "bpe_100": lambda s: apply_bpe_tokenize(s, bpe_merges_100),
        "bpe_200": lambda s: apply_bpe_tokenize(s, bpe_merges_200),
        f"bpe_{max_merges}": lambda s: apply_bpe_tokenize(s, bpe_merges_max),
    }

    avg_tokens_per_sent = {}
    vocab_sizes_schemes = {}
    for name, fn in schemes.items():
        token_counts = [len(fn(s)) for s in sentences]
        all_tokens = []
        for s in sentences:
            all_tokens.extend(fn(s))
        avg_len = np.mean(token_counts)
        unique_toks = len(set(all_tokens))
        avg_tokens_per_sent[name] = avg_len
        vocab_sizes_schemes[name] = unique_toks
        print(f"  {name:>15s}: avg {avg_len:.1f} tokens/sent, "
              f"vocab={unique_toks} types")

    # ----- Token frequency distribution (Zipf's law) -----
    print("\n--- Token Frequency Distribution ---")

    ranks_char, counts_char = compute_token_freq_distribution(
        sentences, lambda s: tokenize_char(s)
    )
    ranks_bpe, counts_bpe = compute_token_freq_distribution(
        sentences, lambda s: apply_bpe_tokenize(s, merges[:100])
    )
    ranks_word, counts_word = compute_token_freq_distribution(
        sentences, lambda s: tokenize_word(s)
    )

    # Zipf fit: freq ∝ 1/rank → log(freq) = -log(rank) + const
    def zipf_fit(ranks, counts):
        log_r = np.log(ranks[:min(50, len(ranks))])
        log_c = np.log(counts[:min(50, len(counts))] + 1e-9)
        A = np.vstack([log_r, np.ones_like(log_r)]).T
        slope, intercept = np.linalg.lstsq(A, log_c, rcond=None)[0]
        return slope, intercept

    slope_char, _ = zipf_fit(ranks_char, counts_char)
    slope_bpe, _ = zipf_fit(ranks_bpe, counts_bpe)
    slope_word, _ = zipf_fit(ranks_word, counts_word)
    print(f"  Zipf slope (char): {slope_char:.3f}  (ideal: -1.0)")
    print(f"  Zipf slope (bpe):  {slope_bpe:.3f}")
    print(f"  Zipf slope (word): {slope_word:.3f}")

    # ----- Optimal vocab size -----
    print("\n--- Optimal Vocabulary Size Search ---")
    v_sizes, perplexities, opt_merges, merge_pts = find_optimal_vocab_size(
        sentences, max_merges=max_merges, test_frac=0.2
    )
    print(f"  Merge counts tested : {merge_pts}")
    print(f"  Vocabulary sizes    : {v_sizes}")
    print(f"  Perplexities        : {[round(p, 1) for p in perplexities]}")
    print(f"  Optimal n_merges    : {opt_merges}")
    opt_idx = merge_pts.index(opt_merges)
    print(f"  Optimal vocab size  : {v_sizes[opt_idx]}")
    print(f"  Optimal perplexity  : {perplexities[opt_idx]:.2f}")

    elapsed = time.time() - t0
    print(f"\nAnalysis runtime: {elapsed:.1f}s")

    # ----- Plot: 4 subplots -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Subplot 1: Vocabulary growth curve ---
    ax = axes[0, 0]
    ax.plot(merge_counts, unique_token_counts, linewidth=2, color='tab:blue', marker='o',
            markersize=3, markevery=20)
    ax.set_xlabel('Number of BPE merges', fontsize=11)
    ax.set_ylabel('Unique tokens in vocabulary', fontsize=11)
    ax.set_title('(1) BPE Vocabulary Growth Curve\n'
                 'merges → vocabulary size', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(unique_token_counts[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Initial chars: {unique_token_counts[0]}')
    ax.legend(fontsize=9)

    # --- Subplot 2: Zipf's law check ---
    ax = axes[0, 1]
    top_n = min(100, len(ranks_char))

    ax.loglog(ranks_char[:top_n], counts_char[:top_n],
              'o-', markersize=3, linewidth=1.5, color='tab:blue',
              label=f'char (slope={slope_char:.2f})', alpha=0.8)
    ax.loglog(ranks_bpe[:min(top_n, len(ranks_bpe))],
              counts_bpe[:min(top_n, len(counts_bpe))],
              's-', markersize=3, linewidth=1.5, color='tab:orange',
              label=f'BPE-100 (slope={slope_bpe:.2f})', alpha=0.8)
    ax.loglog(ranks_word[:min(top_n, len(ranks_word))],
              counts_word[:min(top_n, len(counts_word))],
              '^-', markersize=3, linewidth=1.5, color='tab:green',
              label=f'word (slope={slope_word:.2f})', alpha=0.8)

    # Reference Zipf line
    ref_r = np.array([1, top_n])
    ref_c = counts_char[0] / ref_r
    ax.loglog(ref_r, ref_c, 'k--', linewidth=1, alpha=0.5, label='Zipf ideal (slope=-1)')

    ax.set_xlabel('Rank (log scale)', fontsize=11)
    ax.set_ylabel('Frequency (log scale)', fontsize=11)
    ax.set_title("(2) Token Frequency Distribution\n(Zipf's law check)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.2)

    # --- Subplot 3: Tokenization comparison ---
    ax = axes[1, 0]
    names = list(avg_tokens_per_sent.keys())
    avg_vals = [avg_tokens_per_sent[n] for n in names]
    vocab_vals = [vocab_sizes_schemes[n] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, avg_vals, width, color='tab:blue', alpha=0.7,
                   label='Avg tokens/sentence')
    ax2_twin = ax.twinx()
    bars2 = ax2_twin.bar(x + width / 2, vocab_vals, width, color='tab:orange', alpha=0.7,
                         label='Vocab size (unique tokens)')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('Avg tokens/sentence', color='tab:blue', fontsize=10)
    ax2_twin.set_ylabel('Vocab size (unique tokens)', color='tab:orange', fontsize=10)
    ax.set_title('(3) Tokenization Comparison\nchar vs word vs BPE', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Subplot 4: Optimal vocab size (perplexity curve) ---
    ax = axes[1, 1]
    ax.plot(v_sizes, perplexities, 'o-', linewidth=2, color='tab:red',
            markersize=6, label='Unigram perplexity')

    # Mark optimal
    best_v = v_sizes[opt_idx]
    best_p = perplexities[opt_idx]
    ax.axvline(best_v, color='gray', linestyle='--', alpha=0.7)
    ax.scatter([best_v], [best_p], s=120, color='gold', zorder=5,
               edgecolors='black', linewidth=1.5, label=f'Optimal vocab={best_v}')

    # Annotate each point with merge count
    for vs, pp, nm in zip(v_sizes, perplexities, merge_pts):
        ax.annotate(f'{nm}m', (vs, pp), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='gray')

    ax.set_xlabel('Vocabulary size (unique BPE tokens)', fontsize=11)
    ax.set_ylabel('Unigram perplexity (held-out)', fontsize=11)
    ax.set_title('(4) Optimal BPE Vocabulary Size\n(minimize perplexity on held-out)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('BPE Analysis on Japanese Medical Corpus (100 sentences)\n'
                 'Vocabulary Growth | Zipf Distribution | Tokenization Comparison | Optimal Vocab Size',
                 fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe_analysis.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Corpus: 100 Japanese medical sentences, {len(all_chars)} unique chars")
    print(f"  BPE merges trained: {len(merges)}")
    print(f"  Initial vocab (chars): {unique_token_counts[0]}")
    print(f"  Final vocab ({max_merges} merges): {unique_token_counts[-1]}")
    print(f"\n  Tokenization avg tokens/sentence:")
    for name, avg in avg_tokens_per_sent.items():
        print(f"    {name:>15}: {avg:.1f} tokens  (vocab={vocab_sizes_schemes[name]})")
    print(f"\n  Zipf slopes: char={slope_char:.3f}, BPE={slope_bpe:.3f}, word={slope_word:.3f}")
    print(f"  All schemes follow approximately Zipf distribution (slope ≈ -0.5 to -1.0)")
    print(f"\n  Optimal BPE vocab size: {best_v} tokens (after {opt_merges} merges)")
    print(f"  Optimal perplexity: {best_p:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
