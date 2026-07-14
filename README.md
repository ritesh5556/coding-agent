How Layers Connect Now

Layer 5  Agent class           ── calls ──►  run_agent_loop()
                                                  │
Layer 4  agent_loop._run()  ◄── yields AgentEvent ┘
            │ calls stream_fn ──► Layer 1 groq_stream()  (gets LlmStreamEvent stream)
            │ calls execute_tool_calls ──► Layer 3 executor  (gets ToolResultMessage)
            │
Layer 2  types.py  ── all events/messages/config flow as these types ──
Layer 4 is the conductor: pulls from Layer 1 (LLM), delegates to Layer 3 (tools), re-emits everything as unified AgentEvents for Layer 5 to consume.

Note: I added AgentLoopConfig + QueueMode + hook type aliases to types.py:310-333 — plan referenced them but they didn't exist yet.

Next: Layer 5 — agent.py, the stateful class. Holds transcript, queues, subscribers; wraps run_agent_loop; implements prompt(), steer(), follow_up(), abort(), subscribe(). Ready?