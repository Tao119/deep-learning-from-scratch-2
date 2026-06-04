"""
nlp_applications/knowledge_graph.py — Simple Knowledge Graph + QA

Extracts (subject, relation, object) triples via pattern matching,
builds a graph, and answers factual questions by traversal.
"""
from __future__ import annotations

import re
from collections import defaultdict


# ─────────────────────────────────────── triples ───────────────────────────────

MEDICAL_SCIENCE_TRIPLES: list[tuple[str, str, str]] = [
    # 治療する (treats)
    ("アスピリン", "治療する", "頭痛"),
    ("アスピリン", "治療する", "発熱"),
    ("アスピリン", "治療する", "心筋梗塞"),
    ("ペニシリン", "治療する", "細菌感染症"),
    ("インスリン", "治療する", "糖尿病"),
    ("ステロイド", "治療する", "アレルギー"),
    ("ステロイド", "治療する", "炎症"),
    ("抗菌薬", "治療する", "肺炎"),
    ("抗菌薬", "治療する", "敗血症"),
    ("モルヒネ", "治療する", "疼痛"),
    ("ワクチン", "治療する", "インフルエンザ"),
    ("化学療法", "治療する", "がん"),
    ("透析", "治療する", "腎不全"),
    ("輸血", "治療する", "貧血"),
    ("酸素投与", "治療する", "低酸素症"),
    # 合併症 (complication)
    ("糖尿病", "合併症", "腎症"),
    ("糖尿病", "合併症", "網膜症"),
    ("糖尿病", "合併症", "神経障害"),
    ("糖尿病", "合併症", "心筋梗塞"),
    ("高血圧", "合併症", "脳卒中"),
    ("高血圧", "合併症", "心不全"),
    ("高血圧", "合併症", "腎不全"),
    ("肝硬変", "合併症", "食道静脈瘤"),
    ("肝硬変", "合併症", "腹水"),
    ("動脈硬化", "合併症", "心筋梗塞"),
    ("動脈硬化", "合併症", "脳梗塞"),
    # 機能 (function)
    ("心臓", "機能", "血液循環"),
    ("肺", "機能", "ガス交換"),
    ("肝臓", "機能", "解毒"),
    ("腎臓", "機能", "老廃物除去"),
    ("脳", "機能", "思考"),
    ("膵臓", "機能", "血糖調節"),
    ("甲状腺", "機能", "代謝調節"),
    ("骨髄", "機能", "造血"),
    ("胃", "機能", "消化"),
    ("皮膚", "機能", "体温調節"),
    # 症状 (symptom)
    ("心筋梗塞", "症状", "胸痛"),
    ("肺炎", "症状", "発熱"),
    ("肺炎", "症状", "咳"),
    ("糖尿病", "症状", "口渇"),
    ("貧血", "症状", "疲労感"),
    ("高血圧", "症状", "頭痛"),
    ("アレルギー", "症状", "蕁麻疹"),
    ("骨折", "症状", "疼痛"),
    # 原因 (cause)
    ("喫煙", "原因", "肺がん"),
    ("肥満", "原因", "糖尿病"),
    ("ウイルス", "原因", "インフルエンザ"),
    ("細菌", "原因", "肺炎"),
]


# ─────────────────────────────────────── knowledge graph ───────────────────────

class KnowledgeGraph:
    """Simple in-memory knowledge graph supporting triple store and QA."""

    def __init__(self):
        # (subject, relation) → list of objects
        self._sr2o: dict[tuple[str, str], list[str]] = defaultdict(list)
        # (relation, object) → list of subjects
        self._ro2s: dict[tuple[str, str], list[str]] = defaultdict(list)
        # (subject, object) → list of relations
        self._so2r: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.triples: list[tuple[str, str, str]] = []

    def add_triple(self, subj: str, rel: str, obj: str) -> None:
        key = (subj, rel, obj)
        if key in {(s, r, o) for s, r, o in self.triples}:
            return
        self.triples.append((subj, rel, obj))
        self._sr2o[(subj, rel)].append(obj)
        self._ro2s[(rel, obj)].append(subj)
        self._so2r[(subj, obj)].append(rel)

    def query_objects(self, subj: str, rel: str) -> list[str]:
        """What objects does subj have via rel?"""
        return self._sr2o.get((subj, rel), [])

    def query_subjects(self, rel: str, obj: str) -> list[str]:
        """What subjects have obj via rel?"""
        return self._ro2s.get((rel, obj), [])[:]

    def query_relations(self, subj: str, obj: str) -> list[str]:
        """What relations exist between subj and obj?"""
        return self._so2r.get((subj, obj), [])

    def all_subjects(self) -> set[str]:
        return {s for s, _, _ in self.triples}

    def all_objects(self) -> set[str]:
        return {o for _, _, o in self.triples}

    def all_relations(self) -> set[str]:
        return {r for _, r, _ in self.triples}

    def stats(self) -> dict:
        return {
            "triples": len(self.triples),
            "subjects": len(self.all_subjects()),
            "relations": len(self.all_relations()),
            "objects": len(self.all_objects()),
        }


# ─────────────────────────────────────── triple extraction ─────────────────────

class TripleExtractor:
    """
    Pattern-based triple extractor for Japanese text.
    Supports patterns like:
      - X は Y を 治療する
      - X の 合併症 は Y
      - X は Y の 機能 を 持つ
    """

    PATTERNS = [
        # Pattern: Subject は/が Object を Relation（する）
        (r"([\w・]+)[はが](.+?)を(治療する|治療している|治療します)", 0, 2, 1),
        # Pattern: Subject の Relation は Object
        (r"([\w・]+)の(合併症|症状|原因|機能)[はが]([\w・,、]+)", 0, 1, 2),
        # Pattern: Subject は Object という/の Relation を持つ
        (r"([\w・]+)[はが]([\w・]+)の(機能|働き)を(持つ|担う|担っている)", 0, 2, 1),
        # Simple: Subject は Object を引き起こす
        (r"([\w・]+)[はが]([\w・]+)を(引き起こす|引き起こします|招く)", 0, 2, 1),
    ]

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        triples = []
        for pattern, si, ri, oi in self.PATTERNS:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                try:
                    subj = groups[si].strip()
                    rel = groups[ri].strip()
                    obj_str = groups[oi].strip()
                    for obj in re.split(r"[,、]", obj_str):
                        obj = obj.strip()
                        if obj:
                            triples.append((subj, rel, obj))
                except IndexError:
                    continue
        return triples


# ─────────────────────────────────────── QA engine ─────────────────────────────

class KnowledgeGraphQA:
    """
    Simple QA engine that traverses a KnowledgeGraph to answer questions.

    Supported question types:
      - "Xは何を治療しますか"  → query_objects(X, 治療する)
      - "Xの合併症は何ですか"  → query_objects(X, 合併症)
      - "Xの症状は何ですか"    → query_objects(X, 症状)
      - "誰/何がYを治療しますか" → query_subjects(治療する, Y)
      - "YはXの何ですか"       → query_relations(X, Y)
    """

    QUESTION_PATTERNS = [
        # "Xは何を<Rel>しますか" or "Xの<Rel>は何ですか"
        (r"(.+?)は何を(治療|治癒|予防)(し|でき|し)", "objects", lambda m: (m.group(1), "治療する")),
        (r"(.+?)の(合併症|症状|原因|機能)は何ですか", "objects", lambda m: (m.group(1), m.group(2))),
        (r"(.+?)が(.+?)を(治療|予防)(し|でき)", "subjects", lambda m: (m.group(3) + "する", m.group(2))),
        (r"何が(.+?)を(治療|引き起こ|予防)", "subjects", lambda m: ("治療する" if "治療" in m.group(2) else "原因", m.group(1))),
        (r"(.+?)はどのような(機能|働き)を持ちますか", "objects", lambda m: (m.group(1), "機能")),
        (r"(.+?)の(治療|治療法)は何ですか", "subjects", lambda m: ("治療する", m.group(1))),
    ]

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def answer(self, question: str) -> list[str]:
        """Return list of answer strings, or ['不明'] if no answer found."""
        q = question.strip()

        # Rule 1: "何を治療しますか" → objects of 治療する
        m = re.search(r"^([\w・]+)は何を治療", q)
        if m:
            return self.kg.query_objects(m.group(1), "治療する") or ["不明"]

        # Rule 2: "の合併症は何ですか"
        m = re.search(r"^([\w・]+)の合併症は", q)
        if m:
            return self.kg.query_objects(m.group(1), "合併症") or ["不明"]

        # Rule 3: "の症状は何ですか"
        m = re.search(r"^([\w・]+)の症状は", q)
        if m:
            return self.kg.query_objects(m.group(1), "症状") or ["不明"]

        # Rule 4: "の機能は何ですか"
        m = re.search(r"^([\w・]+)の機能は", q)
        if m:
            return self.kg.query_objects(m.group(1), "機能") or ["不明"]

        # Rule 5: "の原因は何ですか"
        m = re.search(r"^([\w・]+)の原因は", q)
        if m:
            return self.kg.query_objects(m.group(1), "原因") or ["不明"]

        # Rule 6: "何が/誰がXを治療しますか"
        m = re.search(r"([\w・]+)を治療(する|します|できる)", q)
        if m:
            return self.kg.query_subjects("治療する", m.group(1)) or ["不明"]

        # Rule 7: "何/誰がXを引き起こしますか"
        m = re.search(r"([\w・]+)を引き起こ", q)
        if m:
            return self.kg.query_subjects("原因", m.group(1)) or ["不明"]

        # Rule 8: entity → find all relations
        for entity in list(self.kg.all_subjects()) + list(self.kg.all_objects()):
            if entity in q and len(entity) > 1:
                rels = self.kg.query_objects(entity, "治療する")
                if rels:
                    return rels
        return ["不明"]


# ─────────────────────────────────────── benchmark ─────────────────────────────

QA_QUERIES: list[dict] = [
    # objects of 治療する
    {"question": "アスピリンは何を治療しますか？",    "answers": ["頭痛", "発熱", "心筋梗塞"]},
    {"question": "インスリンは何を治療しますか？",    "answers": ["糖尿病"]},
    {"question": "ペニシリンは何を治療しますか？",    "answers": ["細菌感染症"]},
    {"question": "ステロイドは何を治療しますか？",    "answers": ["アレルギー", "炎症"]},
    {"question": "化学療法は何を治療しますか？",      "answers": ["がん"]},
    {"question": "透析は何を治療しますか？",          "answers": ["腎不全"]},
    {"question": "抗菌薬は何を治療しますか？",        "answers": ["肺炎", "敗血症"]},
    # objects of 合併症
    {"question": "糖尿病の合併症は何ですか？",        "answers": ["腎症", "網膜症", "神経障害", "心筋梗塞"]},
    {"question": "高血圧の合併症は何ですか？",        "answers": ["脳卒中", "心不全", "腎不全"]},
    {"question": "肝硬変の合併症は何ですか？",        "answers": ["食道静脈瘤", "腹水"]},
    {"question": "動脈硬化の合併症は何ですか？",      "answers": ["心筋梗塞", "脳梗塞"]},
    # objects of 機能
    {"question": "心臓の機能は何ですか？",            "answers": ["血液循環"]},
    {"question": "肺の機能は何ですか？",              "answers": ["ガス交換"]},
    {"question": "腎臓の機能は何ですか？",            "answers": ["老廃物除去"]},
    {"question": "肝臓の機能は何ですか？",            "answers": ["解毒"]},
    {"question": "骨髄の機能は何ですか？",            "answers": ["造血"]},
    # reverse: who treats X
    {"question": "心筋梗塞を治療するものは何ですか？",  "answers": ["アスピリン"]},
    {"question": "糖尿病を治療するものは何ですか？",    "answers": ["インスリン"]},
    {"question": "がんを治療するものは何ですか？",      "answers": ["化学療法"]},
    {"question": "腎不全を治療するものは何ですか？",    "answers": ["透析"]},
]


def evaluate_qa(qa_engine: KnowledgeGraphQA,
                queries: list[dict]) -> tuple[float, list[dict]]:
    """
    Evaluate QA engine.
    Correct if any predicted answer overlaps with gold answers.
    """
    results = []
    correct = 0
    for q in queries:
        preds = qa_engine.answer(q["question"])
        golds = q["answers"]
        hit = any(p in golds for p in preds)
        correct += int(hit)
        results.append({
            "question": q["question"],
            "predicted": preds,
            "gold": golds,
            "correct": hit,
        })
    accuracy = correct / len(queries) if queries else 0.0
    return accuracy, results


# ─────────────────────────────────────── main ──────────────────────────────────

if __name__ == "__main__":
    # Build graph
    kg = KnowledgeGraph()
    for s, r, o in MEDICAL_SCIENCE_TRIPLES:
        kg.add_triple(s, r, o)

    print("=== Knowledge Graph Statistics ===")
    stats = kg.stats()
    for k, v in stats.items():
        print(f"  {k:10s}: {v}")

    print("\n=== Sample Triples (first 10) ===")
    for s, r, o in MEDICAL_SCIENCE_TRIPLES[:10]:
        print(f"  ({s}, {r}, {o})")

    # Pattern-based extraction demo
    extractor = TripleExtractor()
    sample_texts = [
        "アスピリンは頭痛を治療する。",
        "糖尿病の合併症は腎症、網膜症、神経障害である。",
        "心臓は血液循環の機能を持つ。",
    ]
    print("\n=== Triple Extraction Demo ===")
    for text in sample_texts:
        triples = extractor.extract(text)
        print(f"  Text   : {text}")
        print(f"  Triples: {triples}")

    # QA benchmark
    qa = KnowledgeGraphQA(kg)
    accuracy, results = evaluate_qa(qa, QA_QUERIES)

    print(f"\n=== QA Benchmark: {len(QA_QUERIES)} queries ===")
    print(f"  Accuracy: {accuracy:.3f}  ({sum(r['correct'] for r in results)}/{len(results)})")

    print("\n=== Sample QA Results ===")
    for r in results[:8]:
        status = "✓" if r["correct"] else "✗"
        print(f"  [{status}] Q: {r['question']}")
        print(f"       Pred: {r['predicted']}  Gold: {r['gold']}")

    print("\n=== Manual Queries ===")
    manual = [
        "アスピリンは何を治療しますか？",
        "糖尿病の合併症は何ですか？",
        "心臓の機能は何ですか？",
        "腎不全を治療するものは何ですか？",
        "高血圧の合併症は何ですか？",
    ]
    for q in manual:
        ans = qa.answer(q)
        print(f"  Q: {q}")
        print(f"  A: {ans}")
