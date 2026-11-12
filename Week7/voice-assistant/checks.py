"""Runnable checks for Stages 1, 2, 4 and the Stage 6 measurement.

    python checks.py stage1                 # the eight typed checks with the model in config.yaml
    python checks.py stage1 --model qwen3:8b
    python checks.py stage2                 # speech latency and stop
    python checks.py stage4                 # memory survives a restart
    python checks.py stage6                 # stage 1 on every model in llm.compare_models, writes the README table
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timedelta

from va.agent import Agent
from va.appointments import AppointmentStore
from va.config import load_config, setup_logging, with_overrides


def _next_weekday(now: datetime, weekday: int) -> set[str]:
    """Dates that count as 'on Friday': the next Friday (and the one after if today is Friday)."""
    days = (weekday - now.weekday()) % 7
    out = {(now + timedelta(days=days or 7)).strftime("%Y-%m-%d")}
    if days == 0:
        out.add(now.strftime("%Y-%m-%d"))
    return out


def stage1_cases(now: datetime):
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    fridays = _next_weekday(now, 4)

    def dentist(rows):
        return [r for r in rows if "dentist" in (r["title"] + " " + r["description"]).lower()]

    def c1(before, after, reply, ctx):
        d = dentist(after)
        ok = len(after) == 1 and len(d) == 1 and d[0]["date"] == f"{tomorrow} 15:00" and "chicago" in d[0]["location"].lower()
        if d:
            ctx["dentist_id"] = d[0]["id"]
        return ok, f"rows={after}"

    def c2(before, after, reply, ctx):
        aid = ctx.get("dentist_id")
        return aid is not None and re.search(rf"\b{aid}\b", ctx["reply1"]) is not None, f"id={aid} reply={ctx['reply1']!r}"

    def c3(before, after, reply, ctx):
        return len(after) == len(before) and "?" in reply, f"new rows={len(after) - len(before)}"

    def c4(before, after, reply, ctx):
        return "dentist" in reply.lower(), ""

    def c5(before, after, reply, ctx):
        d = dentist(after)
        ok = len(after) == len(before) and len(d) == 1 and d[0]["id"] == ctx.get("dentist_id") and d[0]["date"] == f"{tomorrow} 16:00"
        return ok, f"rows={after}"

    def c6(before, after, reply, ctx):
        return len(dentist(after)) == 1 and "?" in reply, f"still there={len(dentist(after)) == 1}"

    def c7(before, after, reply, ctx):
        return len(dentist(after)) == 0, f"rows={after}"

    def c8(before, after, reply, ctx):
        new = [r for r in after if r["id"] not in {b["id"] for b in before}]
        ok = (len(new) == 1 and new[0]["date"][:10] in fridays and new[0]["date"].endswith("12:00")
              and "roma" in new[0]["location"].lower())
        return ok, f"new={new}"

    return [
        ("book a dentist appointment tomorrow at 3 pm in Chicago", "one appointment, tomorrow 15:00, Chicago", c1),
        (None, "the id is in the answer", c2),
        ("book a haircut on Friday at 10 am", "nothing saved, the agent asks where", c3),
        ("what is on my calendar", "the dentist appointment is mentioned", c4),
        ("move my dentist appointment to 4 pm", "same appointment, now 16:00, no duplicate", c5),
        ("delete my dentist appointment", "asks first, still there", c6),
        ("yes", "deleted", c7),
        ("book lunch with Sam on Friday at noon at Cafe Roma", "exactly one appointment", c8),
    ]


def run_stage1(cfg: dict, model: str | None = None, verbose: bool = True) -> dict:
    """Type the eight commands into a fresh agent with an empty database and check the stored data."""
    store = AppointmentStore(":memory:")
    agent = Agent(cfg, store, memory=None, model=model)
    agent.warm_up()
    now = datetime.now()
    ctx, rows, passed, total = {}, [], [], 0.0
    for i, (text, expected, check) in enumerate(stage1_cases(now), 1):
        before = store.list()
        reply = ctx.get("reply1", "")
        if text is not None:
            res = agent.run(text)
            total += res.seconds
            reply = res.reply
            if i == 1:
                ctx["reply1"] = reply
            tools = ", ".join(e["tool"] + ("" if e["result"].get("ok") else "!") for e in res.tool_events) or "-"
        else:
            tools = "-"
        ok, detail = check(before, store.list(), reply, ctx)
        passed.append(ok)
        rows.append({"#": i, "typed": text or "(reply)", "expected": expected, "pass": ok,
                     "tools": tools, "reply": reply, "detail": detail})
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'} {i}. {text or '(reply)'}\n     tools: {tools}\n     reply: {reply}")
            if not ok:
                print(f"     {detail}")
    summary = {"model": agent.model, "passed": sum(passed), "seconds": round(total, 1), "rows": rows}
    if verbose:
        print(f"\n{agent.model}: {summary['passed']}/8 passed in {summary['seconds']} s")
    return summary


def run_stage2(cfg: dict) -> dict:
    """Time to first speech for a booking, and how fast stop() silences a sentence."""
    from va.assistant import Assistant
    cfg = with_overrides(cfg, **{"appointments.db_path": "data/check_stage2.db", "memory.extract_in_background": False})
    a = Assistant(cfg, voice=False, speak=True).start()
    a.wait_ready()
    a.store.clear()
    t0 = time.perf_counter()
    tid = a.submit("book a dentist appointment tomorrow at 3 pm in Chicago")
    while a.speaker.first_audio_at is None or a.speaker.first_audio_at < t0:
        time.sleep(0.01)
        if time.perf_counter() - t0 > 60:
            raise TimeoutError("no speech after 60 s")
    first = a.speaker.first_audio_at - t0
    while a.turn(tid)["state"] not in ("done", "error"):
        time.sleep(0.05)
    a.speaker.wait()
    a.speaker.say("This is a long sentence that keeps going and going so that we have time to interrupt it in the middle.")
    time.sleep(1.0)
    proc = a.speaker._proc
    t1 = time.perf_counter()
    a.stop_speaking()
    while proc is not None and proc.poll() is None:
        time.sleep(0.005)
    stop = time.perf_counter() - t1
    print(f"first words after {first:.2f} s (need < 4)   ->  {'PASS' if first < 4 else 'FAIL'}")
    print(f"silent {stop * 1000:.0f} ms after stop()      ->  {'PASS' if stop < 0.5 else 'FAIL'}")
    return {"first_speech_s": round(first, 2), "stop_s": round(stop, 3)}


def run_stage4(cfg: dict) -> bool:
    """Remember a preference, 'restart' (new objects, nothing shared but Qdrant), then ask about it."""
    from va.memory import MemoryStore
    cfg = with_overrides(cfg, **{"memory.collection": cfg["memory"]["collection"] + "_check"})
    MemoryStore(cfg).clear()

    def fresh_agent():
        mem = MemoryStore(cfg)
        return Agent(cfg, AppointmentStore(":memory:"), memory=mem), mem

    agent, mem = fresh_agent()
    r1 = agent.run("remember that I prefer morning meetings")
    print("reply 1:", r1.reply, "| tools:", [e["tool"] for e in r1.tool_events])
    print("memories:", [m["text"] for m in mem.all()])
    del agent, mem                               # the "restart": no history, no objects survive
    agent, mem = fresh_agent()
    r2 = agent.run("what time of day do I prefer for meetings")
    ok = "morning" in r2.reply.lower()
    print("reply 2:", r2.reply, "| memory used:", [m["text"] for m in r2.memories])
    print("PASS" if ok else "FAIL")
    return ok


def write_readme_table(results: list[dict], path="README.md") -> str:
    lines = ["| Model | Checks passed (of 8) | Total seconds for the eight checks |", "|---|---|---|"]
    lines += [f"| {r['model']} | {r['passed']} | {r['seconds']} |" for r in results]
    table = "\n".join(lines)
    text = open(path).read()
    new = re.sub(r"(<!-- stage6 -->\n).*?(\n<!-- /stage6 -->)", lambda m: m.group(1) + table + m.group(2), text, flags=re.S)
    open(path, "w").write(new)
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["stage1", "stage2", "stage4", "stage6"])
    p.add_argument("--model")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.stage == "stage1":
        sys.exit(0 if run_stage1(cfg, args.model)["passed"] == 8 else 1)
    if args.stage == "stage2":
        run_stage2(cfg)
    if args.stage == "stage4":
        run_stage4(cfg)
    if args.stage == "stage6":
        results = [run_stage1(cfg, m) for m in cfg["llm"]["compare_models"]]
        print(write_readme_table(results))
