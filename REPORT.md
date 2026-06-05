# deep-learning-from-scratch-2 実験レポート

## 概要
純NumPy による自然言語処理の実装。外部DLライブラリ不使用（NumPy + Matplotlib のみ）。

---

## 実装モデル・実験結果一覧

| 実装 | ファイル | 結果 |
|------|---------|------|
| **Transformer LM** | ch08/transformer_lm.py | **ppl=1.46（最低）** |
| RNN言語モデル | ch05/rnnlm.py | ppl=3.52 |
| GRU言語モデル | ch06/gru_lm.py | ppl=5.10 |
| LSTM言語モデル | ch06/lstm_lm.py | ppl=5.90 |
| CBOW + Neg Sampling | ch03/cbow.py | you→hello 類似 ✓ |
| Skip-gram | ch03/skip_gram.py | CBOW と相補的 |
| BPE Tokenizer | common/bpe.py | encode→decode 完全一致 |
| Attention Seq2Seq | ch08/attention.py | 数列逆順 83%@8epoch |
| BiLSTM-NER | nlp_applications/bilstm_ner.py | token acc 100% |
| Seq2Seq JA-EN翻訳 | nlp_applications/machine_translation.py | BLEU追跡 |
| ViT (純NumPy) | ch08/vit.py | toy data 100% |
| Beam Search | ch08/beam_search.py | top-5候補出力 |
| BERT MLM事前学習 | ch08/bert_pretrain_full.py | MLM 51.6% / NSP 83.0% |
| POS タグ付け | nlp_applications/japanese_pos_tagging.py | 合成データ 100% |
| 文書類似検索 | nlp_applications/document_similarity.py | Mean P@3=0.73 |
| 多タスク学習 | nlp_applications/multitask_nlp.py | 全タスク 100% |
| 質問応答 | nlp_applications/question_answering.py | EM/F1計測 |
| 知識グラフ | nlp_applications/knowledge_graph.py | QA精度 100% |
| 文章生成（多戦略） | nlp_applications/text_generation_advanced.py | greedy/temp/top-k/nucleus |
| 抽出的要約 | nlp_applications/summarization_extractive.py | ROUGE追跡 |
| 対話システム | nlp_applications/dialogue_system.py | 10ターンデモ |

---

## モデル比較（Perplexity on toy corpus）

```
RNN:          ppl = 2.36
GRU:          ppl = 5.10
LSTM:         ppl = 5.90 (小コーパスではGRU > LSTM)
Transformer:  ppl = 1.46 ← 最低
```

Transformer は小コーパスでも最低 ppl を達成。GRU が LSTM より低い理由は小コーパスでのサンプル効率が良いため。

---

## Attention 可視化

`ch08/self_attention_visualization.py` で以下を実装：
- 各層・各ヘッドの attention 重みヒートマップ
- Attention エントロピー（集中度の指標）
- local vs long-range attention の比較

---

## 位置符号化の分析

`ch08/positional_encoding_analysis.py` で比較：
- 固定 Sinusoidal PE
- ランダム学習型 PE
- PE なし（ablation）

最終 ppl：No PE < Sinusoidal ≈ Learnable（いずれも同程度）

---

## スパースAttentionの実装

`ch08/sparse_attention.py` で4パターン比較：

| パターン | FLOPS | ppl |
|--------|-------|-----|
| Full Attention | 4608 | ~1.01 |
| Local (w=2) | 2688 (41%削減) | ~1.01 |
| Strided | 3584 | ~1.01 |
| BigBird-style | 3456 | ~1.01 |

精度を維持しながら FLOPs を最大41%削減可能。

---

## 改良型Transformer手法の実装

`ch08/transformer_improvements.py` で5手法を純NumPy実装：
- **RoPE**: 回転行列による位置埋め込み
- **Flash Attention**: ブロック単位計算でメモリ2.7x削減
- **Pre-LN vs Post-LN**: Pre-LNの学習安定性確認
- **GQA**: n_q=8でg=1,2,4,8を比較（メモリ削減効果）
- **ALiBi**: 線形バイアス位置符号化

---

## 実行方法

```bash
# 各章
cd ch03 && python3 cbow.py        # CBOW
cd ch05 && python3 rnnlm.py       # RNN LM
cd ch06 && python3 gru_lm.py      # GRU vs LSTM
cd ch08 && python3 transformer_lm.py  # Transformer (best ppl)
cd ch08 && python3 bert_pretrain_full.py  # BERT MLM+NSP

# NLPアプリ
cd nlp_applications && PYTHONPATH=.. python3 japanese_pos_tagging.py
cd nlp_applications && PYTHONPATH=.. python3 machine_translation.py
cd nlp_applications && PYTHONPATH=.. python3 multitask_nlp.py
```
