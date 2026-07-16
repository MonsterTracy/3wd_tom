# Architecture invariants

1. The identity space is always the same 21 unordered player pairs.
2. First-order knowledge masks and second-order output masks are separate APIs.
3. Living-player state never changes identity classes.
4. Environment events are deterministic; only speech is parsed by an LLM.
5. Private facts use the same schema and stream machinery as public events.
6. The belief-label call is independent of gameplay and never receives truth as
   a fallback label.
7. First-order inputs contain only public events, observer-visible private
   events, and observer ID.
8. Public-only second-order inputs contain only public events and target ID.
9. Wolf-conditioned second-order inputs contain modeler-visible events,
   modeler ID, and target ID; no target-private events are included.
10. Training data must pass the exact `tom.v1` schema and contain a successful
    elicitation label permitted by its output mask.

## Data flow

```text
game action
  -> deterministic GAME_EVENT / PRIVATE_FACT
  -> optional speech-local parsed events
  -> public or private state checkpoint
  -> independent target-belief elicitation
  -> successful tom.v1 sample OR separate failure record
  -> event tokenization
  -> Transformer / GRU / bag-of-events MLP
  -> masked 21-pair softmax
  -> pair and seven-player marginal metrics
```
