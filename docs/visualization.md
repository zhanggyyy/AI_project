# Terminal Visualization Design

The terminal layer is for debugging, teaching, and watching agents play. It is
not a graphical game UI.

## Renderer Contract

`TerminalRenderer.render_state(state, viewer=None, reveal=False)` returns a
string. It does not print directly. This makes rendering easy to test and lets
callers decide whether output goes to a terminal, file, notebook, or replay log.

`viewer` controls private information:

- `viewer=None`: public spectator view, all hands hidden.
- `viewer=player_id`: that player's hand is visible.
- `reveal=True`: perfect-information debug view.

## State Display

The initial state renderer should show:

- round number and current phase
- current player
- deck size
- hidden set-aside card as `??` unless reveal mode is active
- each player row with status, protection, hand visibility, and discard pile
- recent public events

Example:

```text
Round 1 | ACTION | Current: P0 Alice | Deck: 10 | Hidden: ??

Players
  > P0 Alice   alive protected: no   hand: Guard, ??   discard: Priest
    P1 Bob     alive protected: yes  hand: ??           discard: Handmaid
    P2 Chen    out   protected: no   hand: --           discard: Princess

Recent Events
  [public] Alice played Priest targeting Bob
  [private:P0] Alice saw Bob's hand
```

Private events should be included only when rendering for a viewer allowed to see
them, or when `reveal=True`.

## Action Display

Legal actions should be rendered as stable numbered choices:

```text
Legal Actions
  0. Play Guard -> Bob, guess Priest
  1. Play Guard -> Bob, guess Baron
  2. Play Priest -> Bob
```

Stable ordering matters for replay, deterministic debugging, and future CLI
manual play.

## Replay Support

Replay should be event-driven. The environment records `HistoryEntry` objects
after setup and after each action. A replay runner can render:

1. state before action
2. legal action list
3. chosen action
4. events emitted by resolution
5. resulting state

For hidden-information debugging, the replay interface should support switching
viewer perspective at each step.

## Debug Modes

- `public`: masks all private information.
- `viewer`: shows one player's own hand and private events.
- `reveal`: shows all hands, hidden card, deck order, and private events.

`reveal` mode is useful for tests and engine debugging, but agent code should not
depend on it.
