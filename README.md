# ゼロから作るDeepLearning② 自然言語処理編

純NumPy実装。外部DLライブラリ不使用。

## 章構成

| 章 | テーマ | 実装 | 精度/結果 |
|----|--------|------|-----------|
| ch01 | ニューラルネットワーク復習 | スパイラル分類 (2層NN) | **96.7%** |
| ch02 | 単語の分散表現 | 共起行列・PPMI・SVD | 2D可視化 |
| ch03 | word2vec (CBOW) | Simple CBOW, Softmax損失 | you→hello類似 |
| ch04 | word2vec高速化 | Negative Sampling, Embedding | 学習安定化 |
| ch05 | RNN・BPTT | SimpleRNNLM, Truncated BPTT | ppl=3.52 |
| ch06 | LSTM・GRU | 2層LSTM + Weight Tying | ppl改善 |
| ch07 | 文章生成 | LSTMで自然な文生成 | パターン学習 |
| ch08 | Attention | Seq2Seq+Attention, 数列逆順タスク | **83%** (8epoch) |

## 実装済みコンポーネント

```
common/
├── layers.py       # Affine, Sigmoid, Relu, SoftmaxWithLoss, MatMul
├── embedding.py    # Embedding, EmbeddingDot, NegativeSamplingLoss, UnigramSampler
├── time_layers.py  # RNN, TimeRNN, LSTM, TimeLSTM, TimeEmbedding, TimeAffine, TimeAttention
├── optimizer.py    # SGD, Adam
├── trainer.py      # Trainer, remove_duplicate
└── util.py         # preprocess, create_co_matrix, ppmi, cos_similarity, most_similar, analogy
```

## 実行方法

```bash
cd ch01 && python3 train_spiral.py       # スパイラル分類
cd ch02 && python3 ppmi_svd.py           # 単語ベクトル可視化
cd ch03 && python3 cbow.py               # SimpleCBOW
cd ch04 && python3 cbow_negative.py      # CBOW + Negative Sampling
cd ch05 && python3 rnnlm.py              # RNN言語モデル
cd ch06 && python3 lstm_lm.py            # LSTM言語モデル
cd ch07 && python3 generate.py           # 文章生成
cd ch08 && python3 attention.py          # Attention Seq2Seq
```
