"""Interactive terminal prototype.

You are always P0 Alice. The terminal stays in Alice's player view: Alice can see
her own hand, public information, and private events meant for Alice only. Other
players are controlled by a random policy.
"""

from __future__ import annotations

from collections import defaultdict

from loveletter_ai.actions import Action
from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.players import PlayerId
from loveletter_ai.renderers import TerminalRenderer
from loveletter_ai.state import Phase


HUMAN = PlayerId(0)


def main() -> None:
    env = LoveLetterEnv(["Alice", "Bob"], seed=77)
    state = env.reset()
    renderer = TerminalRenderer()

    print("Love Letter terminal prototype")
    print("You are P0 Alice. Your view stays fixed for the whole round.")
    print()
    print(renderer.render_state(state, viewer=HUMAN))

    while state.phase != Phase.ROUND_OVER:
        actor = state.current_player
        if actor is None:
            raise RuntimeError("No current player.")

        if actor == HUMAN:
            input("\nYour turn. Press Enter to draw a card...")
            state = env.begin_action_phase()
            if state.phase == Phase.ROUND_OVER:
                break

            print()
            print(renderer.render_state(state, viewer=HUMAN))
            legal_actions = env.legal_actions(HUMAN)
            action = choose_human_action(legal_actions)
            print(f"\nYou chose: {action.label()}")
        else:
            actor_name = player_name(state, actor)
            input(f"\n{actor_name}'s turn. Press Enter to watch...")
            state = env.begin_action_phase()
            if state.phase == Phase.ROUND_OVER:
                break

            legal_actions = env.legal_actions(actor)
            action = env.rng.choice(list(legal_actions))
            print(f"\n{actor_name} chose: {action.label()}")

        state = env.step(action)
        print()
        print(renderer.render_state(state, viewer=HUMAN))

    print_result(state)


def choose_human_action(legal_actions: tuple[Action, ...]) -> Action:
    by_card: dict[CardName, list[Action]] = defaultdict(list)
    for action in legal_actions:
        by_card[action.card].append(action)

    card_options = sorted(by_card, key=lambda card: CARD_SPECS[card].value)
    print("\nYour cards / legal plays")
    for index, card in enumerate(card_options):
        print(f"  {index}. {card.value}")

    card = card_options[prompt_index("Choose a card to play: ", len(card_options))]
    card_actions = by_card[card]

    if len(card_actions) == 1:
        return card_actions[0]

    targets = sorted(
        {action.target for action in card_actions if action.target is not None},
        key=lambda player_id: int(player_id),
    )
    target = None
    if targets:
        print("\nTargets")
        for index, player_id in enumerate(targets):
            print(f"  {index}. P{int(player_id)}")
        target = targets[prompt_index("Choose a target: ", len(targets))]

    target_actions = [
        action
        for action in card_actions
        if target is None or action.target == target
    ]

    guesses = [action.guess for action in target_actions if action.guess is not None]
    if guesses:
        print("\nGuard guesses")
        for index, guess in enumerate(guesses):
            print(f"  {index}. {guess.value}")
        guess = guesses[prompt_index("Choose a guess: ", len(guesses))]
        return next(action for action in target_actions if action.guess == guess)

    if len(target_actions) == 1:
        return target_actions[0]

    print("\nActions")
    for index, action in enumerate(target_actions):
        print(f"  {index}. {action.label()}")
    return target_actions[prompt_index("Choose an action: ", len(target_actions))]


def prompt_index(prompt: str, count: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= choice < count:
            return choice
        print(f"Please enter a number from 0 to {count - 1}.")


def player_name(state, player_id: PlayerId) -> str:
    for player in state.players:
        if player.id == player_id:
            return player.name
    return f"P{int(player_id)}"


def print_result(state) -> None:
    print()
    if state.winner is None:
        print("Round result: no winner")
        return
    winner = next(player for player in state.players if player.id == state.winner)
    if winner.id == HUMAN:
        print(f"Round result: you win as P{int(winner.id)} {winner.name}")
    else:
        print(f"Round result: P{int(winner.id)} {winner.name} wins")


if __name__ == "__main__":
    main()
