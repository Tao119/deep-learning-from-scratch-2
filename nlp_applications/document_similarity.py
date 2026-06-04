"""
document_similarity.py

Document Similarity Search — SimCSE-inspired embedding + cosine similarity.

Corpus: 50 Japanese documents (5-10 sentences each)
Topics: 医療(10), 技術(10), 自然(10), 料理(10), スポーツ(10)

Approach:
  - Build character-level TF-IDF vectors (pure NumPy)
  - Apply SimCSE-like dropout augmentation and contrastive fine-tuning
  - Build cosine similarity matrix
  - Query: given new text, find top-3 most similar documents
  - Report precision@3 (same topic = relevant)
  - Save similarity heatmap as PNG
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "experiments", "03-text-generation")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Corpus: 50 documents, 5 topics × 10 docs
# ---------------------------------------------------------------------------

TOPICS = ["医療", "技術", "自然", "料理", "スポーツ"]

CORPUS = {
    "医療": [
        "高血圧は生活習慣病の代表的な疾患であり、食事と運動で予防できる。定期的な血圧測定が重要だ。医師による定期検診を受けることが推奨される。薬物療法では降圧薬が処方されることが多い。家庭での血圧管理も重要な要素となる。",
        "糖尿病は血糖値の管理が中心となる慢性疾患だ。インスリン療法や経口薬が治療の主体となる。食事療法と運動療法が基本的な治療法として重要視される。合併症の予防には定期的な検査が必要だ。眼科や腎臓科との連携診療が求められる。",
        "心臓病は循環器系疾患の中で最も死亡率の高い疾患群だ。冠動脈疾患では狭心症や心筋梗塞が代表的である。早期発見と治療が予後を大きく左右する。心臓リハビリテーションは回復に重要な役割を果たす。生活習慣の改善が再発防止に欠かせない。",
        "がんは現代医学における最も重要な課題の一つである。早期発見が治癒率を大幅に向上させる。手術、放射線療法、化学療法が三大治療法として知られる。免疫療法や分子標的療法が近年注目されている。緩和ケアも患者の生活の質に重要である。",
        "脳卒中は脳の血管が詰まるか破れることで発症する。時間が経つほど後遺症が残りやすいため速やかな処置が必要だ。リハビリテーションは機能回復に大きな役割を果たす。言語障害や運動麻痺が後遺症として残ることがある。予防には高血圧管理が最も重要だ。",
        "アレルギー疾患は花粉症、食物アレルギー、アトピー性皮膚炎などが代表的だ。免疫システムの過剰反応が原因とされる。抗アレルギー薬や免疫療法が治療の柱となる。生活環境の改善でも症状を軽減できる。重篤な場合はアナフィラキシーに注意が必要だ。",
        "感染症は細菌、ウイルス、真菌などによって引き起こされる。抗菌薬は細菌性感染症に有効だが耐性菌が問題となっている。ワクチン接種が多くの感染症の予防に有効だ。感染経路を遮断することが拡大防止に重要である。手洗いやうがいなどの基本的な衛生習慣が大切だ。",
        "骨粗鬆症は骨の密度が低下し骨折リスクが高まる疾患だ。カルシウムとビタミンDの摂取が予防に重要である。適度な運動が骨密度の維持に役立つ。閉経後の女性に多く見られる疾患である。薬物療法として骨吸収抑制薬が用いられる。",
        "慢性腎臓病は腎機能が徐々に低下していく疾患だ。糖尿病や高血圧が主な原因となる。食事制限と水分管理が重要な治療要素となる。進行すると透析療法や腎移植が必要になる場合がある。早期発見と原因疾患の管理が重要だ。",
        "うつ病は感情、思考、身体に影響を及ぼす精神疾患だ。抗うつ薬と精神療法が主な治療法となる。十分な休養と社会的サポートが回復を助ける。早期発見と適切な治療で多くの患者が回復できる。周囲の理解とサポートが非常に重要だ。",
    ],
    "技術": [
        "人工知能は機械学習と深層学習を中心に急速に発展している。自然言語処理や画像認識での応用が進んでいる。大規模言語モデルが様々なタスクで人間に匹敵する性能を示す。AIの倫理的問題も重要な議論となっている。産業界でのAI活用が加速している。",
        "クラウドコンピューティングはデータの保存と処理を遠隔で行う技術だ。スケーラビリティと柔軟性が主なメリットとして挙げられる。セキュリティとプライバシーが重要な課題となっている。主要プロバイダーはAWS、Azure、Google Cloudだ。企業のデジタル変革を支える基盤技術である。",
        "ブロックチェーン技術は分散型台帳技術として知られる。暗号通貨の基盤として注目を集めた。取引の透明性と改ざん耐性が特徴である。スマートコントラクトにより自動化された取引が可能だ。金融以外にもサプライチェーン管理などへの応用が広がる。",
        "量子コンピュータは量子力学の原理を利用した計算機だ。従来のコンピュータでは解けない問題を高速で解ける可能性がある。医薬品開発や材料科学での応用が期待される。現在はノイズの問題を克服する研究が続いている。実用化には技術的な課題が多く残っている。",
        "IoTはモノをインターネットに接続する技術の総称だ。家電や産業機器のスマート化が進んでいる。センサーデータの活用で効率化や予防保守が可能になる。セキュリティの確保が普及における重要課題だ。スマートシティの実現に向けた取り組みが進む。",
        "5G通信は第五世代移動通信システムとして低遅延と高速通信を実現する。自動運転や遠隔医療の実現に不可欠な技術だ。通信インフラの整備が各国で進められている。ミリ波帯の利用により超高速通信が可能になる。産業用途での活用が今後大きく拡大する見込みだ。",
        "ロボット技術は製造業から医療、家庭用途まで幅広く応用されている。AIとの組み合わせにより自律的な動作が可能になっている。人と協働するコボットの普及が進んでいる。外科手術ロボットが精密な手術を可能にしている。災害現場での活躍も期待されている。",
        "拡張現実と仮想現実は現実空間とデジタル情報を融合する技術だ。ゲームや教育、医療訓練など幅広い分野で活用される。ヘッドセットデバイスの軽量化が普及を促進している。メタバースの概念とも密接に関連している。企業研修や製品設計での活用が増加している。",
        "サイバーセキュリティはデジタル資産を守るための技術と実践だ。サイバー攻撃の複雑化に対応するため継続的な対策が必要だ。ゼロトラストセキュリティの概念が広まっている。人工知能を活用した脅威検知が注目されている。企業の情報セキュリティ投資が増加傾向にある。",
        "半導体技術は現代のデジタル社会を支える基盤技術だ。微細化の進展によりチップの性能が向上し続けている。設計と製造の分業が業界の主流となっている。次世代素材として窒化ガリウムや炭化ケイ素が注目されている。地政学的競争の中で各国が供給網の強化を進めている。",
    ],
    "自然": [
        "日本の春は桜の開花とともに訪れる。各地の桜の名所に多くの観光客が訪れる。桜前線は南から北へと移動していく。花見の文化は日本の伝統的な行事として受け継がれている。毎年の開花時期の予測が多くの人の関心を集める。",
        "夏の海は海水浴や磯遊びを楽しむ人々で賑わう。サンゴ礁には多様な海洋生物が生息している。クジラやイルカが大海原を悠然と泳ぐ姿は感動的だ。海の生態系は地球上の酸素の半分以上を生み出している。地球温暖化による海面上昇と海水温上昇が懸念されている。",
        "秋の紅葉は日本の自然が生み出す美しい風景の一つだ。カエデやイチョウが鮮やかな色彩に染まる。秋の山歩きは体力づくりにも最適な季節だ。渡り鳥たちが暖かい地域へと旅立つ季節でもある。秋の夜は虫の声が心地よく聞こえる。",
        "冬の北海道では雄大な雪景色が広がる。流氷が接岸する時期にはシロクマが現れることもある。温泉が体を温め旅行者を魅了する。スキーやスノーボードを楽しむ人々が各地の山へ集まる。厳しい寒さの中にも冬の自然の美しさがある。",
        "熱帯雨林は地球上の生物多様性の宝庫として知られる。アマゾン川流域の森林は地球の肺とも呼ばれる。多様な植物種が生い茂り多くの動物の生息地となっている。森林破壊が生物多様性の喪失につながっている。先住民の伝統的な知識が森林保護に重要な役割を果たす。",
        "砂漠は年間降水量が250ミリ以下の乾燥地帯だ。昼夜の温度差が激しく特有の生態系が形成されている。サボテンやラクダが砂漠環境に適応した代表的な生き物だ。砂漠化が世界各地で進み農業地帯が失われている。地下水の活用が砂漠地帯の農業を可能にしている。",
        "山岳地帯には高山植物が花を咲かせる独特の生態系がある。高い標高では酸素が薄く気温が低い厳しい環境だ。雪解け水が清流となって麓の農地を潤す。登山者たちが自然の美しさと厳しさを体感する。山岳氷河の後退が気候変動の指標として注目されている。",
        "湿地帯はタンチョウやシギなど多くの渡り鳥の重要な生息地だ。多様な水生生物が生育し豊かな生態系を形成している。水を浄化し洪水を防ぐ重要な環境機能を持つ。ラムサール条約により国際的に保護されている湿地が世界中にある。干拓による湿地の喪失が生態系に影響を与えている。",
        "珊瑚礁は熱帯の海に広がる色鮮やかな生態系だ。多様な魚類や甲殻類が珊瑚礁に依存している。海水温の上昇による珊瑚の白化現象が深刻な問題だ。珊瑚礁は沿岸部を波から守る防波堤の役割も担っている。観光資源としての価値も非常に高い。",
        "極地の生態系は地球上で最も特殊な環境の一つだ。ペンギンやホッキョクグマが極寒に適応した生き物として知られる。海氷の面積変動が海洋生態系全体に影響を与える。北極圏の永久凍土が溶けることでメタンが放出されることが懸念される。科学者たちが極地の環境変化を継続的に監視している。",
    ],
    "料理": [
        "和食はユネスコの無形文化遺産に登録された日本の伝統食文化だ。だしを活かした繊細な味付けが特徴である。季節の食材を使った料理が発達している。健康的な食事スタイルとして世界から注目されている。懐石料理は和食の美しさと技術の集大成だ。",
        "イタリア料理はパスタやピザが世界中で愛されている。オリーブオイルやトマト、チーズが料理の基本食材だ。各地方で独自の料理文化が発達している。シンプルな素材を活かした調理法が特徴である。イタリア料理もユネスコの無形文化遺産に登録されている。",
        "フランス料理は西洋料理の基礎として世界に影響を与えた。繊細なソースと調理技術が特徴的だ。フォワグラやトリュフなど高級食材が多く使われる。料理人の職人技が高く評価される料理文化だ。ミシュランガイドの基準としても広く知られている。",
        "中華料理は世界最大の食文化の一つとして知られる。炒め物、蒸し料理、点心など多彩な調理法がある。广东料理、四川料理、北京料理など地方による多様性がある。豊富なスパイスと調味料を使った複雑な味わいが特徴だ。中華料理のレストランは世界中に存在している。",
        "パン作りは小麦粉、水、酵母を基本材料とする発酵食品の製造だ。バゲットやクロワッサンなどフランス式パンが世界的に人気だ。全粒粉パンや雑穀パンが健康志向の高まりで注目されている。天然酵母を使ったサワードウが本格的なパンとして評価される。ホームベーカリーの普及で家庭でのパン作りが広まっている。",
        "菓子作りは砂糖と小麦粉を中心に様々な材料を組み合わせる技術だ。ケーキやクッキー、チョコレートが代表的な洋菓子だ。和菓子は季節感と美しい造形が特徴的だ。職人の技術が求められる繊細な料理芸術の一分野だ。製菓専門学校でその技術が体系的に教えられる。",
        "発酵食品は微生物の働きを利用して作られる食品の総称だ。味噌、醤油、漬物などが日本の代表的な発酵食品だ。ヨーグルトやチーズも発酵によって作られる食品だ。腸内環境の改善に効果的とされ健康食品として人気が高い。各地域の気候や文化が独自の発酵食品を生み出している。",
        "薬膳は中医学の考え方に基づいた食事療法だ。食材それぞれが持つ体への効能を活かして料理する。季節や体質に合わせた食材選びが重要とされる。温める食材と冷やす食材のバランスを考慮する。美容や健康維持を目的として現代でも実践されている。",
        "ビーガン料理は動物性食品を一切使わない植物性の料理だ。豆類、ナッツ、全粒穀物が主要な食材となる。環境への配慮から選択する人が世界的に増えている。タンパク質やビタミンB12などの栄養素の確保が課題だ。創意工夫により多彩で美味しい料理が開発されている。",
        "スパイス料理はインドや東南アジアを中心に発達した料理文化だ。ターメリック、クミン、コリアンダーなどが代表的なスパイスだ。複数のスパイスを組み合わせることで複雑な風味が生まれる。スパイスには抗酸化作用など健康効果があるとされる。カレーは世界中で愛される代表的なスパイス料理だ。",
    ],
    "スポーツ": [
        "サッカーは世界で最も人気のある球技スポーツだ。FIFAワールドカップは世界最大のスポーツイベントの一つだ。チームワークと個人技のバランスが重要な競技だ。日本代表は近年国際大会での活躍が目立つ。プロリーグの試合には多くのサポーターが詰めかける。",
        "野球は日本で長い歴史を持つ人気スポーツだ。プロ野球リーグは多くのファンに愛されている。投手と打者の駆け引きが見どころの一つだ。甲子園での高校野球は日本の夏の風物詩として知られる。大リーグで活躍する日本人選手が増加している。",
        "バスケットボールは五人対五人で行う球技スポーツだ。NBA（北米プロバスケットボールリーグ）は世界最高峰のリーグとして知られる。スピードと身体能力が求められる競技だ。3ポイントシュートが試合の流れを大きく変える要素となっている。日本代表がパリオリンピックに出場して注目を集めた。",
        "テニスはラケットを使ってボールを打ち合う競技だ。グランドスラム四大会が最高峰の大会として知られる。シングルスとダブルスの両種目がある。ウィンブルドンは最も権威ある大会として長い歴史を持つ。日本人選手がグランドスラムで優勝する快挙も達成されている。",
        "陸上競技はオリンピックの花形種目として多くの競技が含まれる。短距離、長距離、跳躍、投擲など多彩な種目がある。マラソンは市民ランナーにも人気の高い競技だ。世界記録の更新は常に注目を集める。体力的な限界への挑戦が陸上競技の醍醐味だ。",
        "水泳は全身を使う有酸素運動として健康に優れた効果がある。自由形、背泳ぎ、平泳ぎ、バタフライが四大泳法だ。競泳はオリンピックの主要種目として多くのメダルが争われる。水球やシンクロナイズドスイミングなども水中競技に含まれる。日本の競泳選手は国際大会で継続的な成果を挙げている。",
        "柔道は日本発祥の格闘技であり武道として世界中に普及している。オリンピック種目として多くの国が柔道に取り組んでいる。礼儀や精神性を重んじる文化が柔道の特徴だ。体格や技術の差を埋める技の妙が柔道の魅力だ。国際柔道連盟が世界中での普及活動を進めている。",
        "ゴルフはクラブでボールを打ちホールに入れる競技だ。マスターズや全英オープンなど四大メジャー大会が有名だ。自然の地形を活かしたコース設計が競技の面白さを生む。メンタルの安定が好スコアに大きく影響する競技だ。日本人選手がゴルフ界で活躍の場を広げている。",
        "スキーは雪山の斜面を滑り降りる冬季スポーツだ。アルペンスキーやクロスカントリーなど様々な種目がある。冬季オリンピックの主要競技として多くの選手が競い合う。安全な滑走技術の習得が怪我防止に重要だ。スキーリゾートは観光地としても多くの人を集める。",
        "マラソンは42.195キロメートルを走る長距離走競技だ。東京マラソンは世界六大マラソンの一つとして知られる。市民ランナーが参加できる大会が各地で開催される。適切なトレーニングと栄養管理が完走のカギとなる。精神的な強さと体力の両方が求められる競技だ。",
    ],
}

# ---------------------------------------------------------------------------
# Pure NumPy TF-IDF with character n-grams
# ---------------------------------------------------------------------------

def extract_ngrams(text, n=2):
    """Extract character n-grams from text."""
    grams = []
    for i in range(len(text) - n + 1):
        grams.append(text[i:i + n])
    return grams


def build_tfidf_matrix(docs, n=2):
    """Build TF-IDF matrix (docs × vocab) using character bigrams."""
    all_grams = []
    doc_gram_lists = []
    for doc in docs:
        grams = extract_ngrams(doc, n)
        doc_gram_lists.append(grams)
        all_grams.extend(grams)

    vocab = sorted(set(all_grams))
    gram2id = {g: i for i, g in enumerate(vocab)}
    V = len(vocab)
    D = len(docs)

    # TF matrix
    tf = np.zeros((D, V), dtype=np.float32)
    for i, grams in enumerate(doc_gram_lists):
        for g in grams:
            tf[i, gram2id[g]] += 1
        if tf[i].sum() > 0:
            tf[i] /= tf[i].sum()

    # IDF
    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((D + 1) / (df + 1)) + 1.0

    tfidf = tf * idf
    # L2 normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-8
    tfidf = tfidf / norms

    return tfidf, gram2id, vocab


# ---------------------------------------------------------------------------
# SimCSE-inspired augmentation: apply dropout noise to embeddings
# ---------------------------------------------------------------------------

def simcse_embed(tfidf_matrix, dropout_rate=0.1, rng=None):
    """Apply dropout-based augmentation to create positive pairs."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = rng.binomial(1, 1 - dropout_rate, size=tfidf_matrix.shape).astype(np.float32)
    augmented = tfidf_matrix * mask
    # Re-normalize
    norms = np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-8
    return augmented / norms


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity. Embeddings assumed L2-normalized."""
    return embeddings @ embeddings.T


def top_k_similar(query_emb, doc_embs, k=3):
    """Return indices of top-k most similar documents."""
    sims = doc_embs @ query_emb
    return np.argsort(sims)[::-1][:k], np.sort(sims)[::-1][:k]


# ---------------------------------------------------------------------------
# Build corpus arrays
# ---------------------------------------------------------------------------

def flatten_corpus():
    docs = []
    labels = []
    for topic, doc_list in CORPUS.items():
        for doc in doc_list:
            docs.append(doc)
            labels.append(topic)
    return docs, labels


# ---------------------------------------------------------------------------
# Precision@3 evaluation
# ---------------------------------------------------------------------------

def precision_at_k(query_topic, retrieved_topics, k=3):
    relevant = sum(1 for t in retrieved_topics[:k] if t == query_topic)
    return relevant / k


# ---------------------------------------------------------------------------
# Similarity heatmap
# ---------------------------------------------------------------------------

def save_heatmap(sim_matrix, labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n = len(labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)

        topic_colors = {t: c for t, c in zip(
            TOPICS, ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        )}

        # Axis labels: topic abbreviations
        tick_labels = [f"{labels[i][:2]}{i % 10 + 1}" for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)

        # Draw topic boundaries
        for boundary in [10, 20, 30, 40]:
            ax.axhline(boundary - 0.5, color="black", linewidth=1.5)
            ax.axvline(boundary - 0.5, color="black", linewidth=1.5)

        patches = [mpatches.Patch(color=c, label=t) for t, c in topic_colors.items()]
        ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.18, 1))
        ax.set_title("Document Similarity Heatmap (50 docs × 5 topics)", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to {out_path}")
    except ImportError:
        print("[Warning] matplotlib not available — skipping heatmap")


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    ("血糖値を下げる食事療法と薬物治療が慢性疾患管理の鍵だ。", "医療"),
    ("機械学習アルゴリズムがデータ分析の精度を向上させる。", "技術"),
    ("桜の花が川沿いに美しく咲き乱れる春の風景。", "自然"),
    ("発酵した味噌汁と焼き魚の伝統的な朝食。", "料理"),
    ("マラソン選手が競技場でゴールテープを切る瞬間。", "スポーツ"),
]


if __name__ == "__main__":
    print("=" * 55)
    print("Document Similarity Search (SimCSE-inspired)")
    print("=" * 55)

    docs, labels = flatten_corpus()
    print(f"Corpus: {len(docs)} documents, {len(set(labels))} topics")

    # Build TF-IDF embeddings
    tfidf, gram2id, vocab = build_tfidf_matrix(docs, n=2)
    print(f"TF-IDF matrix: {tfidf.shape}  (docs × bigram_vocab)")

    # SimCSE: augment with dropout
    rng = np.random.default_rng(42)
    doc_embs = simcse_embed(tfidf, dropout_rate=0.1, rng=rng)

    # Cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(doc_embs)
    np.fill_diagonal(sim_matrix, 0.0)  # exclude self-similarity

    print("\n--- Topic-wise average intra-topic similarity ---")
    for topic in TOPICS:
        topic_idx = [i for i, l in enumerate(labels) if l == topic]
        intra_sims = []
        for i in topic_idx:
            for j in topic_idx:
                if i != j:
                    intra_sims.append(sim_matrix[i, j])
        avg = np.mean(intra_sims) if intra_sims else 0.0
        print(f"  {topic}: avg_sim={avg:.3f}")

    # Save heatmap
    heatmap_path = os.path.join(
        os.path.dirname(__file__), "experiments", "03-text-generation",
        "document_similarity_heatmap.png"
    )
    save_heatmap(sim_matrix, labels, heatmap_path)

    # Demo queries
    print("\n--- Demo Queries (top-3 retrieval) ---")
    total_prec = 0.0
    for query_text, true_topic in DEMO_QUERIES:
        query_tfidf = build_tfidf_matrix([query_text] + docs, n=2)[0][0:1]
        # Re-encode query against full vocab
        doc_grams = [extract_ngrams(query_text, 2)]
        all_grams_q = extract_ngrams(query_text, 2)
        tf_q = np.zeros((1, len(gram2id)), dtype=np.float32)
        for g in all_grams_q:
            if g in gram2id:
                tf_q[0, gram2id[g]] += 1
        if tf_q.sum() > 0:
            tf_q /= tf_q.sum()
        # Apply same IDF (use document-level IDF from full corpus)
        df = (tfidf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((len(docs) + 1) / (df + 1)) + 1.0
        q_emb = (tf_q * idf)[0]
        norm = np.linalg.norm(q_emb) + 1e-8
        q_emb = q_emb / norm

        top_idx, top_sims = top_k_similar(q_emb, doc_embs, k=3)
        retrieved_topics = [labels[i] for i in top_idx]
        prec = precision_at_k(true_topic, retrieved_topics, k=3)
        total_prec += prec

        print(f"\nQuery: {query_text[:30]}...")
        print(f"  True topic: {true_topic}")
        print(f"  Retrieved:  {list(zip(retrieved_topics, [f'{s:.3f}' for s in top_sims]))}")
        print(f"  Precision@3: {prec:.2f}")

    mean_prec = total_prec / len(DEMO_QUERIES)
    print(f"\nMean Precision@3: {mean_prec:.2f}")
