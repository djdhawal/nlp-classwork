"""Terminal front end.

    python cli.py            # type commands, hear the replies
    python cli.py --voice    # always-on: speak to it ("Loop, ..."), Ctrl-C to quit
    python cli.py --quiet    # type commands, print replies only
"""
import argparse
import time

from va.assistant import Assistant
from va.config import load_config

p = argparse.ArgumentParser()
p.add_argument("--voice", action="store_true", help="listen on the microphone")
p.add_argument("--quiet", action="store_true", help="do not speak")
p.add_argument("--model", help="override llm.model from config.yaml")
p.add_argument("--config")
args = p.parse_args()

a = Assistant(load_config(args.config), voice=args.voice, speak=not args.quiet, model=args.model).start()
print("loading models...")
a.wait_ready()
for note in a.notes:
    print("note:", note)
print(f"ready: model={a.agent.model}")

if args.voice:
    print(f'Say "{a.cfg["assistant"]["name"]}, ..." Ctrl-C to quit.')
    seen_heard, seen_turns, last_status = 0, set(), None
    try:
        while True:
            snap = a.snapshot()
            if snap["status"] != last_status:
                print(f"[{snap['status']}]")
                last_status = snap["status"]
            for h in a.heard[seen_heard:]:
                print(f"  heard: {h['text']!r} -> {h['action']}")
            seen_heard = len(a.heard)
            for t in snap["turns"]:
                if t["state"] == "done" and t["id"] not in seen_turns:
                    seen_turns.add(t["id"])
                    for ev in t["tools"]:
                        print(f"  tool: {ev['tool']}({ev['arguments']}) -> ok={ev['result'].get('ok')}")
                    print(f"  reply: {t['reply']}  ({t['seconds']} s)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
else:
    try:
        while True:
            text = input("you> ").strip()
            if not text:
                continue
            if text in ("quit", "exit"):
                break
            t = a.run_text(text)
            for ev in t["tools"]:
                print(f"  tool: {ev['tool']}({ev['arguments']}) -> {ev['result']}")
            print(f"loop> {t['reply']}   ({t['seconds']} s)")
            a.speaker.wait()
    except (KeyboardInterrupt, EOFError):
        pass
