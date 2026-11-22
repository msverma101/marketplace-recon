# Video 2 narration (3:00)

Read at a calm pace; each line starts at the time shown. The prompter (`scripts/uc2_voice.sh`) shows the same lines in sync with the video.


## 1 · Pitch

- **0:00** Every marketplace has the same 3 a.m. problem: orders drop, and nobody knows why yet.
- **0:06** Is it a data glitch, a quieter market, or something broken in checkout?
- **0:12** Today a person digs through dashboards, logs and traces to find out, while orders keep failing.
- **0:19** We built an engine that answers that question on its own, and an agent that acts on the answer, safely.

## 2 · How it works

- **0:30** The engine reads the business through read-only ports: telemetry from the services, and data from the order pipeline.
- **0:38** Every question goes through the same seven stages: frame it, audit the inputs, pick only the methods that apply,
- **0:45** fetch the evidence, evaluate, challenge its own assumptions, and publish a report. Every step lands in a ledger.
- **0:53** The engine itself cannot change anything. On top sits an agent: it turns findings into proposals with an honest range,
- **1:02** and it acts only after a person approves, and only inside a sandbox.

## 3 · Demo (live run)

- **1:10** This is a real, running shop, with 200 products from a Brazilian marketplace and shoppers simulated from real orders.
- **1:17** Left: the world and what the agent sees. Right: the engine at work, every three hours.
- **1:23** Now, without telling the agent, we break payments: one in four fails.
- **1:35** As traffic picks up, the engine raises an alarm: checkout errors far above budget.
- **1:41** Checkout attempts are normal, 50 against 60. The failures sit in the payment service.
- **1:45** The engine itself may not act; its policy gate says so.
- **1:49** The agent proposes switching the faulty flag off: about 37 more orders today, 23 to 55.
- **1:53** A person approves. The change runs in the shop's flag service and is read back.
- **2:04** Payments recover in the next block.

## 4 · Close

- **2:30** So the agent found the cause, ruled out a drop in demand, and stated the expected gain before acting.
- **2:38** It fixed it with one approval, and with zero writes to production.
- **2:47** Next: connect it to our own telemetry, with the same guardrails. That step needs your go-ahead.
