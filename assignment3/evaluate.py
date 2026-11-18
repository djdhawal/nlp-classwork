"""Section 3.3: faithfulness evaluation with DeepEval, gold questions, and the improved pipeline.

The judge is any "provider:name" model from config (Ollama by default), wrapped as a DeepEval
custom model so no paid API key is needed.
"""

import hashlib
import json
import os
import re

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # must be set before deepeval loads

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from . import config

JUDGE_MAX_TOKENS = 2048  # truth/claim lists can be longer than a normal answer
JUDGE_CACHE = config.CACHE_DIR / "judge"


# --------------------------------------------------------------------------- judge LLM


def _client(model):
    """OpenAI-compatible client and bare model name for a "provider:name" string."""
    provider, name = model.split(":", 1)
    p = config.PROVIDERS[provider]
    key = os.environ.get(p["key_env"], "") if p["key_env"] else "ollama"
    return OpenAI(base_url=p["base_url"], api_key=key or "missing-key"), name


def _complete(model, messages, json_mode=False, max_tokens=JUDGE_MAX_TOKENS):
    """One chat completion at temperature 0, cached on disk (re-running the notebook is free)."""
    key = hashlib.sha1(json.dumps([model, messages, json_mode, max_tokens]).encode()).hexdigest()
    path = JUDGE_CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["reply"]
    client, name = _client(model)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=name, messages=messages, temperature=0, max_tokens=max_tokens, **kwargs
    )
    reply = resp.choices[0].message.content or ""
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reply": reply}))
    return reply


def _extract_json(text):
    """Cut the outermost {...} out of a reply (models sometimes add prose or code fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(text[start : end + 1])
    # Small models write "Yes" instead of "yes"; DeepEval's verdict fields are lower-case literals.
    for v in data.get("verdicts", []) if isinstance(data.get("verdicts"), list) else []:
        if isinstance(v, dict) and isinstance(v.get("verdict"), str):
            v["verdict"] = v["verdict"].strip().lower()
    return data


def _skeleton(node, defs):
    """Example instance of a JSON schema node, e.g. {"verdicts": [{"verdict": "yes | no"}]}.

    Small models copy a raw JSON schema back instead of filling it, so they get an example instead.
    """
    if "$ref" in node:
        node = defs[node["$ref"].split("/")[-1]]
    if "anyOf" in node:
        node = next(n for n in node["anyOf"] if n.get("type") != "null")
    if "enum" in node:
        return " | ".join(map(str, node["enum"]))
    t = node.get("type")
    if t == "object":
        return {k: _skeleton(v, defs) for k, v in node.get("properties", {}).items()}
    if t == "array":
        return [_skeleton(node.get("items", {}), defs)]
    return f"<{t or 'value'}>"


def _json_reply(model, prompt, schema=None):
    """Ask for JSON (optionally matching a pydantic schema), retry once on bad output."""
    hint = "\n\nReturn only one valid JSON object, no other text."
    if schema is not None:
        js = schema.model_json_schema()
        hint += " Use exactly this shape (fill in the values):\n" + json.dumps(
            _skeleton(js, js.get("$defs", {}))
        )
    messages = [{"role": "user", "content": prompt + hint}]
    for attempt in range(2):
        reply = _complete(model, messages, json_mode=True)
        try:
            data = _extract_json(reply)
            return schema.model_validate(data) if schema is not None else data
        except (ValueError, ValidationError) as e:
            if attempt == 1:
                raise ValueError(f"judge returned bad JSON twice: {e}\n{reply[:300]}") from e
            messages = messages + [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That was not valid ({str(e)[:200]}). Reply again with "
                    "only the corrected JSON object, and keep every "
                    "reason under 20 words.",
                },
            ]


class JudgeLLM(DeepEvalBaseLLM):
    """DeepEval custom model over any OpenAI-compatible endpoint (Ollama, OpenAI, OpenRouter)."""

    def __init__(self, model=config.JUDGE_LLM):
        # Do not call super().__init__: it would strip the "provider:" prefix we need.
        self.name = model
        self.model = self.load_model()

    def load_model(self):
        return self.name

    def generate(self, prompt, schema=None, **kwargs):
        if schema is None:
            return _complete(self.name, [{"role": "user", "content": prompt}])
        return _json_reply(self.name, prompt, schema)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.name


def faithfulness(question, answer, hits, judge=None):
    """DeepEval faithfulness of `answer` against the retrieved chunk texts -> (score, reason).

    `judge` is a JudgeLLM, a "provider:name" string, or None for config.JUDGE_LLM.
    """
    contexts = [h["text"] for h in hits]
    if not contexts:
        return np.nan, "no retrieved context (pipeline abstained before generation)"
    judge = JudgeLLM(judge) if isinstance(judge, str) else judge or JudgeLLM()
    metric = FaithfulnessMetric(
        model=judge, threshold=0.5, include_reason=True, async_mode=False, verbose_mode=False
    )
    case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
    metric.measure(case, _show_indicator=False)
    return float(metric.score), metric.reason


# --------------------------------------------------------------------------- gold triples


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def make_gold(articles, gold):
    """Gold triples as a DataFrame. `gold` is a list of dicts with article_id, question, answer.

    Adds the article title and `answer_in_article`: True when the answer text appears in the
    article (case and whitespace ignored), so typos in hand-written answers are caught early.
    """
    by_id = articles.set_index("article_id")
    rows = []
    for g in gold:
        art = by_id.loc[g["article_id"]]
        rows.append(
            {
                "article_id": g["article_id"],
                "title": art["title"],
                "question": g["question"],
                "answer": g["answer"],
                "answer_in_article": _norm(g["answer"]) in _norm(art["text"]),
            }
        )
    df = pd.DataFrame(rows)
    bad = df.loc[~df["answer_in_article"], "article_id"].tolist()
    if bad:
        print(f"Check these answers, the text is not in the article: {bad}")
    return df


class _GoldQA(BaseModel):
    question: str
    answer: str


def suggest_gold(articles, n=5, model=config.LLM, min_chars=1500, seed=config.SEED):
    """Pick n long articles from different sources and ask the LLM for one factual question each.

    Only a starting point: read the suggestions and edit the GOLD list by hand.
    """
    pool = articles[articles["text"].str.len() >= min_chars].sample(frac=1, random_state=seed)
    picked = pool.groupby("source", sort=False).head(1).head(n)  # one per source first
    if len(picked) < n:
        picked = pd.concat([picked, pool.drop(picked.index).head(n - len(picked))])
    gold = []
    for _, art in picked.iterrows():
        prompt = (
            "Read this news article and write ONE factual question that can only be answered from "
            "this article (name the specific people, places or organisations so it is not "
            "generic). "
            "The question ends with '?'. The answer must be a short phrase (a name, number, "
            "date or "
            "place) copied word for word from the article.\n\n"
            f"Title: {art['title']}\n\n{art['text'][:6000]}"
        )
        qa = _json_reply(model, prompt, _GoldQA)
        gold.append(
            {"article_id": art["article_id"], "question": qa.question, "answer": qa.answer}
        )
    return make_gold(articles, gold)


# --------------------------------------------------------------------------- evaluation


def run_eval(gold, pipeline_fn, name="baseline", judge=None, use_cache=True):
    """Run every gold question through `pipeline_fn` and score faithfulness.

    `pipeline_fn(question)` returns a dict like `rag.answer` (keys `answer`, `hits`). The table has
    one row per question plus a final MEAN row, and is saved to RESULTS_DIR/eval_<name>.csv.
    """
    config.ensure_dirs()
    path = config.RESULTS_DIR / f"eval_{name}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path)
    judge = JudgeLLM(judge) if isinstance(judge, str) else judge or JudgeLLM()
    rows = []
    for g in gold.to_dict("records") if isinstance(gold, pd.DataFrame) else gold:
        out = pipeline_fn(g["question"])
        hits = out.get("hits") or []
        try:
            score, reason = faithfulness(g["question"], out["answer"], hits, judge)
        except ValueError as e:  # judge kept returning broken JSON: record, move on
            score, reason = np.nan, f"judge error: {str(e)[:200]}"
        rows.append(
            {
                "article_id": g["article_id"],
                "question": g["question"],
                "gold_answer": g["answer"],
                "generated_answer": out["answer"],
                "retrieved_chunks": "; ".join(h["chunk_id"] for h in hits),
                "source_retrieved": int(g["article_id"] in {h["article_id"] for h in hits}),
                "faithfulness": score,
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows)
    mean = {
        "article_id": "MEAN",
        "question": "",
        "source_retrieved": df["source_retrieved"].mean(),
        "faithfulness": df["faithfulness"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def compare(baseline, improved, labels=("baseline", "improved")):
    """Side-by-side faithfulness per question, the delta, and a bar plot (saved to RESULTS_DIR)."""
    b, i = (d[d["article_id"] != "MEAN"].reset_index(drop=True) for d in (baseline, improved))
    table = pd.DataFrame(
        {
            "article_id": b["article_id"],
            "question": b["question"],
            labels[0]: b["faithfulness"],
            labels[1]: i["faithfulness"],
        }
    )
    table["delta"] = table[labels[1]] - table[labels[0]]
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "article_id": "MEAN",
                        "question": "",
                        labels[0]: table[labels[0]].mean(),
                        labels[1]: table[labels[1]].mean(),
                        "delta": table["delta"].mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x, w = np.arange(len(table)), 0.38
    ax.bar(x - w / 2, table[labels[0]], w, label=labels[0], color="#9aa5b1")
    ax.bar(x + w / 2, table[labels[1]], w, label=labels[1], color="#2a6fdb")
    ax.set_xticks(x, [f"Q{k + 1}" for k in range(len(table) - 1)] + ["mean"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("faithfulness")
    ax.set_title("DeepEval faithfulness: baseline vs improved")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(config.RESULTS_DIR / "faithfulness_compare.png", dpi=150)
    plt.show()
    return table


# --------------------------------------------------------------------------- improvements

_ABBREV = re.compile(r"(\b[A-Z]\.|\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|No|vs|Gen|Sen|Rep|Gov)\.)$")


def split_sentences(text):
    """Sentence splitter. Citation tags like [A0001-C02 | Title] stay with their sentence."""
    from .prompts import CITATION_RE

    tags = [m.group(0) for m in CITATION_RE.finditer(text)]
    for n, tag in enumerate(tags):  # hide tags so dots in titles do not split
        text = text.replace(tag, f"\x00{n}\x00", 1)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|(?<=\x00)\s+(?=[A-Z])|\n+", text.strip())
    out = []
    for p in (p.strip() for p in parts):
        if not p:
            continue
        if out and _ABBREV.search(out[-1]):  # "U.S." or "Dr." is not a sentence end
            out[-1] += " " + p
        else:
            out.append(p)
    for n, tag in enumerate(tags):
        out = [p.replace(f"\x00{n}\x00", tag) for p in out]
    return out


def grounding_check(answer, hits, model=config.JUDGE_LLM):
    """Post-generation check: keep only the answer sentences the passages support.

    Each sentence is sent to the judge with the retrieved passages; unsupported ones are dropped.
    Returns {"answer", "kept", "dropped"}.
    """
    from .prompts import CITATION_RE
    from .rag import REFUSAL

    if not hits or answer.strip().startswith(REFUSAL.strip()):
        return {"answer": answer, "kept": [answer], "dropped": []}
    passages = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
    kept, dropped = [], []
    for sent in split_sentences(answer):
        claim = CITATION_RE.sub("", sent).strip()
        if len(claim) < 3:  # a stray citation tag on its own line
            kept.append(sent)
            continue
        prompt = (
            "Passages:\n" + passages + "\n\nSentence: " + claim + "\n\nIs every fact in the "
            "sentence directly stated in the passages above? Answer 'yes' only if the passages "
            "support it; general wording like 'the article says' is fine. "
            'Return JSON: {"supported": "yes" or "no"}'
        )
        data = _json_reply(model, prompt)
        (
            kept if str(data.get("supported", "no")).strip().lower().startswith("y") else dropped
        ).append(sent)
    text = " ".join(kept) if kept else REFUSAL
    return {"answer": text, "kept": kept, "dropped": dropped}


def baseline_pipeline(**kwargs):
    """`rag.answer` with its default settings (extra kwargs passed through, e.g. collection)."""
    from . import rag

    return lambda q: rag.answer(q, **kwargs)


def improved_pipeline(rerank=True, strict=True, ground=True, judge=config.JUDGE_LLM, **kwargs):
    """Re-ranking + strict prompt + post-generation grounding check, as one pipeline function."""
    from . import rag

    def run(question):
        out = rag.answer(question, rerank=rerank, strict=strict, **kwargs)
        if ground:
            g = grounding_check(out["answer"], out["hits"], judge)
            out = {**out, "answer": g["answer"], "dropped_sentences": g["dropped"]}
        return out

    return run
