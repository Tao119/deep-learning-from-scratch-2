"""
nlp_applications/text_clustering.py — Text Clustering with K-means

Cluster 200 Japanese documents across 5 topics using:
  - BoW TF-IDF vectorization (pure NumPy, character n-grams)
  - K-means clustering (pure NumPy)
  - Evaluation: Adjusted Rand Index, Silhouette Score
  - Visualization: 2D PCA scatter colored by cluster

Topics: 科学, 政治, スポーツ, 音楽, 料理
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "07-text-clustering")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

TOPICS = ["科学", "政治", "スポーツ", "音楽", "料理"]
N_PER_TOPIC = 40  # 40 × 5 = 200 documents


# ===========================================================================
# Dataset — 200 synthetic Japanese documents (40 per topic)
# ===========================================================================

_DOCS_RAW: dict[str, list[str]] = {
    "科学": [
        "量子コンピュータは量子ビットを使い、古典コンピュータより高速に特定問題を解く。",
        "ダークマターは宇宙質量の27%を占めるが直接検出はまだされていない。",
        "CRISPR-Cas9はゲノム編集を可能にした革命的な技術だ。",
        "光合成は植物が太陽光をエネルギーに変換するプロセスである。",
        "熱力学第二法則によれば孤立系のエントロピーは増大する。",
        "ブラックホールは光さえも脱出できない強力な重力場を持つ。",
        "DNAの二重らせんはワトソンとクリックが1953年に発見した。",
        "E=mc²は質量とエネルギーの等価性を表すアインシュタインの式だ。",
        "抗生物質は細菌の細胞壁合成を阻害して感染症を治療する。",
        "超伝導体は特定温度以下で電気抵抗がゼロになる物質である。",
        "AIの深層学習は多層ニューラルネットワークを用いる技術だ。",
        "火星の大気は二酸化炭素が主成分で大気圧は地球の1%しかない。",
        "タンパク質はアミノ酸がペプチド結合でつながった高分子だ。",
        "気候変動の主原因は人間活動による温室効果ガスの増加とされる。",
        "素粒子物理学の標準模型は17種の基本粒子を説明する理論体系だ。",
        "海洋深層循環は地球全体の熱輸送に重要な役割を担っている。",
        "ナノテクノロジーは原子・分子スケールで物質を操作する技術だ。",
        "電磁波は電場と磁場が互いに垂直に振動しながら伝播する波だ。",
        "免疫システムは自然免疫と獲得免疫の二層構造を持つ。",
        "宇宙膨張はハッブル定数で記述され現在も加速膨張が続いている。",
        "核融合反応は水素同位体が融合してヘリウムになり莫大なエネルギーを放出する。",
        "遺伝子発現はDNAがmRNAに転写されタンパク質に翻訳されるプロセスだ。",
        "統計力学は多粒子系の巨視的性質を確率論で記述する分野だ。",
        "磁気共鳴画像法は核スピン共鳴を利用した断層撮影技術だ。",
        "進化論は自然選択による種の変化をダーウィンが提唱した理論だ。",
        "光電効果はアインシュタインが説明した量子力学的現象である。",
        "酵素は化学反応の活性化エネルギーを下げる生体触媒だ。",
        "超弦理論は素粒子を点ではなく弦として記述する統一理論候補だ。",
        "海馬は記憶形成に不可欠な脳の部位として知られている。",
        "グラフェンは優れた電気伝導性を持つ二次元炭素材料だ。",
        "地球の年齢は放射性同位体の崩壊から約46億年と推定される。",
        "バイオインフォマティクスはゲノムデータ解析に機械学習を活用する分野だ。",
        "気象予測は流体力学と熱力学の方程式を数値的に解くことで行われる。",
        "有機化学では炭素の4価結合を基礎として多様な分子が形成される。",
        "量子暗号は光子の量子状態を利用した理論上盗聴不可能な通信だ。",
        "脳コンピュータインターフェースは神経活動を解読してコンピュータを操作する。",
        "生物多様性の喪失は生態系サービスに深刻な影響を与える。",
        "ナビエストークス方程式は粘性流体の運動を記述する基本方程式系だ。",
        "再生可能エネルギーは化石燃料依存を減らすために不可欠だ。",
        "合成生物学は生命の設計原理を工学的に応用する新興分野だ。",
    ],
    "政治": [
        "国際連合は1945年に設立された世界最大の国際機関だ。",
        "民主主義は市民が政治的意思決定に参加する政治体制だ。",
        "経済制裁は外交政策の手段として特定国に経済的制限を課す。",
        "憲法改正には国民投票や議会の特別多数が必要な場合が多い。",
        "外交交渉は国家間の利害調整を平和的手段で行うプロセスだ。",
        "選挙制度の設計は民主主義の質を大きく左右する重要な要素だ。",
        "国家予算の配分は政府の優先順位を反映する政治的決定だ。",
        "条約は国家間の合意を法的に拘束する形で文書化したものだ。",
        "人権保護は現代国際法の根幹をなす重要な原則である。",
        "核抑止論は冷戦期から続く国際安全保障の基本概念だ。",
        "気候変動対策は国際的な協調が必要な地球規模の政治課題だ。",
        "移民政策は多くの国で重要な政治的争点となっている。",
        "国際刑事裁判所は戦争犯罪や人道に対する罪を裁く機関だ。",
        "政党政治は現代民主主義の主要な政治組織形態である。",
        "市民社会は政府と市場の間に位置する第三のセクターだ。",
        "国家主権は近代国際秩序の基本原則として確立されている。",
        "安全保障理事会は国連の主要意思決定機関として機能している。",
        "民族自決の原則は第一次世界大戦後に国際政治に広まった。",
        "政治腐敗は民主主義への信頼を損なう深刻な問題だ。",
        "国際貿易協定は各国の経済的利益を調整する重要な仕組みだ。",
        "社会保障制度は国民の基本的生活を保障するための政策体系だ。",
        "三権分立は立法・行政・司法の権力分散を図る原則だ。",
        "地政学は地理的要因が国家の政治的行動に与える影響を研究する。",
        "国際開発援助は途上国の発展を支援する先進国の政策だ。",
        "核不拡散条約は核兵器の拡散を防ぐ国際的な枠組みだ。",
        "選挙公正性の確保は民主主義の正当性の基盤となる。",
        "難民条約は迫害を逃れた人々の保護を定めた国際条約だ。",
        "政府の透明性と説明責任は現代民主主義の重要な価値だ。",
        "欧州連合は欧州統合の深化を目指す地域機構の代表例だ。",
        "冷戦は米ソ両大国のイデオロギー対立による国際緊張だった。",
        "国家安全保障戦略は国家の防衛・外交の基本方針を示す文書だ。",
        "政治哲学は正義・権力・自由に関する根本的な問いを探求する。",
        "多国間主義は複数の国家が協調して国際問題を解決する方式だ。",
        "ロビイング活動は特定の政策決定に影響を与えようとする実践だ。",
        "世界貿易機関は多国間貿易ルールを管理する国際機関だ。",
        "ポピュリズムは既存政治体制への不満を背景に台頭する政治現象だ。",
        "食糧安全保障は国際政治における重要な政策課題となっている。",
        "気候変動はパリ協定を通じて国際的な取り組みが進められている。",
        "政治的分極化は多くの民主主義国で深刻な課題となっている。",
        "国際法は国家間の関係を規律するルールの体系として機能する。",
    ],
    "スポーツ": [
        "サッカーワールドカップは4年に一度開催される世界最大のスポーツイベントだ。",
        "陸上競技の100m走では9秒台後半が世界トップレベルとされる。",
        "野球のピッチャーは球速や変化球の精度が勝利の鍵となる。",
        "バスケットボールのNBAは世界最高レベルのプロリーグだ。",
        "テニスのグランドスラムは全豪・全仏・ウィンブルドン・全米の4大会だ。",
        "水泳の自由形では身体のローリングを活かした泳ぎが速い。",
        "マラソンは42.195kmを走る長距離ランニング競技だ。",
        "体操競技では難度と美しさの両立が高得点の秘訣となる。",
        "柔道はフランスで特に人気が高い日本発祥の格闘技だ。",
        "ゴルフのマスターズはオーガスタで行われる名門トーナメントだ。",
        "ラグビーワールドカップは世界規模の格闘球技の祭典だ。",
        "バレーボールはサーブ・レシーブ・アタックの連係が重要だ。",
        "卓球は反射神経と戦術が求められる日本の得意競技だ。",
        "スキーの大回転では急斜面を高速で滑るコントロールが問われる。",
        "フィギュアスケートは技術点と演技点の合計で競われる採点競技だ。",
        "ボクシングは階級制を設けて体重差による不公平をなくしている。",
        "陸上の高跳びは背面跳びが現在の主流となっている。",
        "水泳のバタフライは全泳法中最も消耗の大きい泳ぎだ。",
        "サイクリングのツール・ド・フランスは世界最難関の自転車レースだ。",
        "相撲は体重や身長よりも技術と瞬発力が勝敗を左右する。",
        "競馬は馬の能力と騎手の技術が組み合わさるスポーツだ。",
        "バドミントンのスマッシュは時速400kmを超えることがある。",
        "アメリカンフットボールはQBのパス能力が攻撃の核心だ。",
        "野球の守備ではグラブさばきと送球の正確さが重要だ。",
        "水上スキーはボートに引かれながら水面を滑る水上スポーツだ。",
        "サッカーのキーパーはゴール前の守護神として重要な役割を担う。",
        "陸上三段跳びはホップ・ステップ・ジャンプの三動作からなる。",
        "バスケットボールの3ポイントシュートは現代戦術の中核だ。",
        "テニスのサービスエースは相手に触れさせないサーブだ。",
        "ボートレースはエンジンを使った日本独自のモータースポーツだ。",
        "スポーツ科学は選手のパフォーマンス向上を科学的に支援する。",
        "オリンピックは4年に一度開催される世界最大の総合スポーツ大会だ。",
        "パラリンピックは障害を持つアスリートが競う国際的な大会だ。",
        "スポーツ栄養学は競技力向上のための食事科学として発展している。",
        "チームスポーツでは個人技と協調性のバランスが重要だ。",
        "スポーツ心理学はメンタルトレーニングで競技力向上を目指す。",
        "運動生理学は身体の仕組みからトレーニングを科学的に考える。",
        "プロスポーツは興行としての側面も持つエンターテインメントだ。",
        "スポーツ傷害予防はリハビリとともに重要な医科学分野だ。",
        "スタジアムの熱狂的な雰囲気はホームアドバンテージを生む。",
    ],
    "音楽": [
        "クラシック音楽はバロックから現代まで数百年の歴史を持つ。",
        "ジャズはアフリカ系アメリカ人の文化から生まれた即興音楽だ。",
        "ロックミュージックはエレキギターとドラムを中心とした音楽だ。",
        "ポップミュージックは大衆に広く親しまれる商業的な音楽だ。",
        "ヒップホップはラップとビートボックスを基盤とした音楽文化だ。",
        "クラシックのオーケストラは弦・管・打楽器で構成される大規模編成だ。",
        "ピアノは88鍵を持ち幅広い音域を一人で演奏できる楽器だ。",
        "ギターはポップスからクラシックまで幅広いジャンルで使われる。",
        "声楽はオペラからポップスまで様々なスタイルがある。",
        "音楽理論は和声・対位法・形式論を学ぶ基礎的な教育内容だ。",
        "電子音楽はシンセサイザーを使ったポスト工業化時代の音楽だ。",
        "民族音楽は各地域の文化的アイデンティティを反映している。",
        "ブルースはアメリカ南部で発展した12小節形式の音楽だ。",
        "レゲエはジャマイカ発祥のリズムと社会的メッセージを持つ音楽だ。",
        "バッハはバロック音楽の最高峰として知られる作曲家だ。",
        "ベートーヴェンは古典派からロマン派への橋渡しをした作曲家だ。",
        "モーツァルトは幼少期から天才を発揮した音楽の神童だった。",
        "録音技術の発展は音楽産業を根本的に変革した。",
        "ストリーミングサービスは音楽消費の形を変えている。",
        "音楽フェスティバルは野外で多くのアーティストが演奏する催しだ。",
        "リズムとメロディーと和声が音楽の三大要素とされる。",
        "編曲はオリジナル曲を異なる演奏形態に書き換える技術だ。",
        "即興演奏はジャズやブルースで重要な演奏技術だ。",
        "音楽教育は子供の認知発達に良い影響を与えるとされる。",
        "コンサートホールの音響設計は演奏の質を大きく左右する。",
        "弦楽四重奏はバイオリン2本・ヴィオラ・チェロの編成だ。",
        "交響曲は通常4楽章からなるオーケストラのための大曲だ。",
        "ソナタ形式は提示部・発展部・再現部からなる古典的な形式だ。",
        "ドレミファソラシドはイタリア語由来の階名唱法に使われる。",
        "メトロノームは一定のテンポを刻む練習用の道具だ。",
        "作曲ソフトウェアはデジタル時代の音楽制作を支えている。",
        "音楽プロデューサーはアーティストのレコーディングを監督する。",
        "ライブパフォーマンスは録音では伝わらない熱量を生む。",
        "ミュージカルは音楽・歌・演技を組み合わせた舞台芸術だ。",
        "伝統的な邦楽は三味線・琴・尺八などを使った音楽だ。",
        "ラップの歌詞は社会的なメッセージを含むことが多い。",
        "音楽療法は音楽を使ったリハビリや精神的ケアの分野だ。",
        "アンサンブルは複数の奏者が協調して演奏する形式だ。",
        "音階は特定の音の並びで西洋音楽では長短2種が基本だ。",
        "コードは複数の音を同時に鳴らした和音のことだ。",
    ],
    "料理": [
        "和食は旨味を大切にする日本伝統の料理文化だ。",
        "フランス料理は世界の料理芸術の模範として称えられる。",
        "イタリア料理はパスタやピザが代表的な地中海料理だ。",
        "中華料理は炒め・蒸し・揚げなど多彩な調理法を持つ。",
        "タイ料理はナンプラーやスパイスを使った複雑な風味が特徴だ。",
        "寿司は魚介類を使った日本を代表する料理だ。",
        "ラーメンはスープと麺の組み合わせで全国に地域特色がある。",
        "天ぷらは衣をつけて揚げる和食の代表的な揚げ物料理だ。",
        "刺身は新鮮な魚介を生で食べる日本固有の料理文化だ。",
        "焼き鳥は鶏肉を串に刺して焼いた日本の定番料理だ。",
        "カレーは多くのスパイスを組み合わせたインド発祥の料理だ。",
        "バーベキューは炭火や直火でじっくり焼く料理スタイルだ。",
        "パエリアはサフランを使ったスペインの米料理だ。",
        "タコスはメキシコ発祥のトルティーヤを使った料理だ。",
        "デザートは食事の最後に食べる甘い料理や菓子の総称だ。",
        "スープは野菜や肉をじっくり煮込んだ汁物料理だ。",
        "サラダは生野菜を中心にドレッシングで和えた料理だ。",
        "グリル料理は食材をあぶったり焼いたりする調理法だ。",
        "蒸し料理は水蒸気で食材をやわらかく仕上げる調理法だ。",
        "発酵食品は微生物の働きで保存性と風味が高まった食品だ。",
        "出汁はかつおや昆布から取る日本料理の基本的なスープだ。",
        "醤油・味噌・みりんは和食の基本調味料として欠かせない。",
        "フュージョン料理は異なる国の料理スタイルを融合させた新しい料理だ。",
        "食材の旬を活かした料理は素材の味を最大限に引き出す。",
        "低温調理は肉の旨味を逃さないモダンな調理法だ。",
        "ベーキングは小麦粉と卵とバターを使って焼く西洋的な調理だ。",
        "ベジタリアン料理は肉を使わずに植物性食材だけで作る料理だ。",
        "ヴィーガン料理は動物性食品を一切使用しない完全植物性料理だ。",
        "食品保存技術は冷蔵・冷凍・乾燥・塩蔵など多様な方法がある。",
        "分子ガストロノミーは科学を応用した革新的な料理技術だ。",
        "スパイスは料理に風味と香りを加える植物由来の調味料だ。",
        "オリーブオイルは地中海料理に欠かせない植物油だ。",
        "ハーブは料理の風味付けや飾りに使われる芳香植物だ。",
        "パン作りは発酵と焼き上げのタイミングが品質を左右する。",
        "魚介料理は新鮮さが最も重要な要素となる料理カテゴリーだ。",
        "野菜の切り方は料理の見た目と食感に大きく影響する。",
        "料理の盛り付けは視覚的な美しさも料理の重要な要素だ。",
        "食文化は地域の気候・歴史・宗教を反映した重要な文化的要素だ。",
        "食育は子供たちに食の大切さを伝える教育活動だ。",
        "栄養バランスの取れた食事は健康維持の基本となる。",
    ],
}

# Flatten to list
ALL_DOCS: list[str] = []
ALL_LABELS: list[int] = []
for topic_idx, topic in enumerate(TOPICS):
    docs = _DOCS_RAW[topic][:N_PER_TOPIC]
    ALL_DOCS.extend(docs)
    ALL_LABELS.extend([topic_idx] * len(docs))

ALL_LABELS_ARR = np.array(ALL_LABELS, dtype=np.int32)
N_DOCS = len(ALL_DOCS)
K = len(TOPICS)

assert N_DOCS == 200, f"Expected 200 docs, got {N_DOCS}"


# ===========================================================================
# TF-IDF Vectorizer (character bigrams + unigrams)
# ===========================================================================

def _tokenize(text: str) -> list[str]:
    """Character unigrams + bigrams."""
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return chars + bigrams


def build_tfidf(docs: list[str], min_df: int = 2) -> tuple[np.ndarray, dict[str, int]]:
    """Build TF-IDF matrix (n_docs × vocab_size)."""
    N = len(docs)

    # Count term frequencies per document
    raw_counts: list[dict[str, int]] = []
    df_counts: dict[str, int] = {}
    for doc in docs:
        toks = _tokenize(doc)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        raw_counts.append(counts)
        for t in set(counts):
            df_counts[t] = df_counts.get(t, 0) + 1

    # Build vocab (filter by min_df)
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df_counts.items() if c >= min_df))}
    V = len(vocab)

    # Compute TF-IDF
    mat = np.zeros((N, V), dtype=np.float32)
    for i, counts in enumerate(raw_counts):
        total = sum(counts.values())
        for tok, cnt in counts.items():
            if tok in vocab:
                mat[i, vocab[tok]] = cnt / total

    # IDF
    df_vec = np.zeros(V, dtype=np.float32)
    for tok, idx in vocab.items():
        df_vec[idx] = df_counts.get(tok, 0)
    idf = np.log((N + 1) / (df_vec + 1)) + 1.0

    mat = mat * idf

    # L2 normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-10)

    return mat, vocab


# ===========================================================================
# K-means (pure NumPy)
# ===========================================================================

class KMeans:
    """Pure NumPy K-means with k-means++ initialization."""

    def __init__(self, k: int = 5, max_iter: int = 300, tol: float = 1e-4,
                 random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization."""
        n = X.shape[0]
        first_idx = int(self.rng.integers(n))
        centroids = [X[first_idx]]
        for _ in range(1, self.k):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = dists / (dists.sum() + 1e-10)
            idx = int(self.rng.choice(n, p=probs))
            centroids.append(X[idx])
        return np.array(centroids, dtype=np.float32)

    def fit(self, X: np.ndarray) -> "KMeans":
        centroids = self._init_centroids(X)
        labels = np.zeros(len(X), dtype=np.int32)

        for it in range(self.max_iter):
            # Assignment step
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            # Update step
            new_centroids = np.zeros_like(centroids)
            for c in range(self.k):
                members = X[new_labels == c]
                new_centroids[c] = members.mean(axis=0) if len(members) > 0 else centroids[c]

            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            labels = new_labels
            self.n_iter_ = it + 1

            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        # Compute inertia
        self.inertia_ = float(sum(
            np.linalg.norm(X[labels == c] - centroids[c], axis=1).sum() ** 2
            for c in range(self.k) if np.sum(labels == c) > 0
        ))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(dists, axis=1).astype(np.int32)


# ===========================================================================
# Evaluation metrics (pure NumPy)
# ===========================================================================

def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index without sklearn."""
    n = len(labels_true)
    assert len(labels_pred) == n

    # Contingency table
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    true_to_idx = {c: i for i, c in enumerate(classes_true)}
    pred_to_idx = {c: i for i, c in enumerate(classes_pred)}
    for t, p in zip(labels_true, labels_pred):
        contingency[true_to_idx[t], pred_to_idx[p]] += 1

    # C(nij, 2) sums
    def comb2(n):
        return n * (n - 1) // 2

    sum_comb_c = sum(comb2(contingency[i, j])
                     for i in range(contingency.shape[0])
                     for j in range(contingency.shape[1]))

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_comb_a = sum(comb2(a) for a in row_sums)
    sum_comb_b = sum(comb2(b) for b in col_sums)
    n_choose_2 = comb2(n)

    expected = sum_comb_a * sum_comb_b / (n_choose_2 + 1e-10)
    max_index = (sum_comb_a + sum_comb_b) / 2

    ari = (sum_comb_c - expected) / (max_index - expected + 1e-10)
    return round(float(ari), 4)


def silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 500) -> float:
    """Approximate silhouette score (subsample for speed)."""
    n = len(X)
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sub, lab_sub = X[idx], labels[idx]
    else:
        X_sub, lab_sub = X, labels

    unique_labels = np.unique(lab_sub)
    if len(unique_labels) <= 1:
        return 0.0

    silhouettes = []
    for i in range(len(X_sub)):
        xi = X_sub[i]
        ci = lab_sub[i]

        # Intra-cluster distance (a)
        same = X_sub[lab_sub == ci]
        if len(same) > 1:
            a = float(np.linalg.norm(same - xi, axis=1).sum() / (len(same) - 1))
        else:
            a = 0.0

        # Inter-cluster distance (b) — min over other clusters
        b_vals = []
        for c in unique_labels:
            if c == ci:
                continue
            other = X_sub[lab_sub == c]
            if len(other) > 0:
                b_vals.append(float(np.linalg.norm(other - xi, axis=1).mean()))
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        silhouettes.append((b - a) / denom if denom > 0 else 0.0)

    return round(float(np.mean(silhouettes)), 4)


# ===========================================================================
# PCA (pure NumPy)
# ===========================================================================

def pca_2d(X: np.ndarray) -> np.ndarray:
    """Reduce to 2D via PCA."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / len(X)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, order[:2]]
    return X_centered @ top2


# ===========================================================================
# Visualization
# ===========================================================================

def plot_clusters(X_2d: np.ndarray, labels: np.ndarray, true_labels: np.ndarray,
                  topic_names: list[str], save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

        for ax, plot_labels, title in [
            (axes[0], true_labels, "True Labels"),
            (axes[1], labels, "K-means Clusters"),
        ]:
            for ci in range(len(TOPICS)):
                mask = plot_labels == ci
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=colors[ci], label=topic_names[ci],
                           alpha=0.7, s=30, edgecolors="white", linewidth=0.3)
            ax.set_title(f"{title} (PCA 2D)", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.3)

        plt.suptitle("Text Clustering — TF-IDF + K-means (K=5)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Cluster plot saved: {save_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot")


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_clustering(verbose: bool = True) -> dict:
    if verbose:
        print(f"Building TF-IDF for {N_DOCS} documents …")
    X, vocab = build_tfidf(ALL_DOCS, min_df=2)
    if verbose:
        print(f"  Vocab size : {len(vocab)}")
        print(f"  Matrix     : {X.shape}")

    if verbose:
        print(f"Running K-means (K={K}) …")
    km = KMeans(k=K, max_iter=300, random_state=42)
    km.fit(X)
    labels = km.labels_

    ari = adjusted_rand_index(ALL_LABELS_ARR, labels)
    sil = silhouette_score(X, labels)

    if verbose:
        print(f"  Iterations : {km.n_iter_}")
        print(f"  Inertia    : {km.inertia_:.2f}")
        print(f"\nEvaluation:")
        print(f"  Adjusted Rand Index : {ari:.4f}")
        print(f"  Silhouette Score    : {sil:.4f}")

        # Show cluster composition
        print("\nCluster composition (true label distribution per cluster):")
        for ci in range(K):
            mask = labels == ci
            true_in_ci = ALL_LABELS_ARR[mask]
            from collections import Counter
            counts = Counter(true_in_ci.tolist())
            top = max(counts, key=counts.get)
            print(f"  Cluster {ci}: {dict(counts)} → majority={TOPICS[top]}")

    # PCA & plot
    if verbose:
        print("\nReducing to 2D with PCA …")
    X_2d = pca_2d(X)

    plot_path = os.path.join(OUT_DIR, "text_clustering.png")
    plot_clusters(X_2d, labels, ALL_LABELS_ARR, TOPICS, plot_path)

    results = {
        "n_docs": N_DOCS,
        "vocab_size": len(vocab),
        "k": K,
        "kmeans_iterations": km.n_iter_,
        "inertia": round(float(km.inertia_), 2),
        "adjusted_rand_index": ari,
        "silhouette_score": sil,
        "cluster_sizes": {int(c): int(np.sum(labels == c)) for c in range(K)},
    }

    out_path = os.path.join(OUT_DIR, "clustering_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\nResults saved to {out_path}")

    return results


# ===========================================================================
# __main__
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Text Clustering — TF-IDF + K-means (200 docs, K=5)")
    print("=" * 60)
    results = run_clustering(verbose=True)
    print("\nFinal summary:")
    print(f"  ARI       : {results['adjusted_rand_index']:.4f}")
    print(f"  Silhouette: {results['silhouette_score']:.4f}")
