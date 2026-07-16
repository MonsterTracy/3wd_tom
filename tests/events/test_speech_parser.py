from werewolf.events.speech_parser import SpeechEventParser


class SequenceBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return next(self.responses)


def test_speech_parser_repairs_once_and_emits_only_local_families():
    backend = SequenceBackend(
        [
            "not json",
            '{"events":[{"event_family":"BELIEF_ASSERTION","target":[2],'
            '"content":{"kind":"CAMP","value":"Werewolf"},'
            '"qualifier":{"polarity":"positive","certainty":"normal"},'
            '"ref_event_id":null,"source_span":"2 is a wolf",'
            '"parser_confidence":0.9}]}'
        ]
    )
    result = SpeechEventParser(backend, "parser").parse(
        utterance="I think 2 is a wolf", utterance_id="u1", day=1,
        phase="speech", turn=4, speaker=1
    )
    assert result.status == "ok"
    assert result.attempts == 2
    assert result.events[0]["event_family"] == "BELIEF_ASSERTION"
    assert result.events[0]["source_type"] == "speech_parser"
    assert result.events[0]["content"] == {"kind": "CAMP", "value": "Werewolf"}
    assert result.events[0]["source_span"] == "2 is a wolf"


def test_speech_parser_drops_forbidden_or_unanchored_output():
    response = (
        '{"events":[{"event_family":"GAME_EVENT","target":[2],'
        '"content":{"kind":"HIDDEN_INTENT","value":true},"qualifier":{},'
        '"ref_event_id":null,"source_span":"made up","parser_confidence":1.0}]}'
    )
    result = SpeechEventParser(SequenceBackend([response, response])).parse(
        utterance="hello", utterance_id="u1", day=1,
        phase="speech", turn=4, speaker=1
    )
    assert result.status == "failed"
    assert result.events == ()
    assert result.attempts == 2


def test_speech_parser_rejects_free_role_values_instead_of_encoding_unknown():
    response = (
        '{"events":[{"event_family":"BELIEF_ASSERTION","target":[2],'
        '"content":{"kind":"ROLE","value":"Wizard"},"qualifier":{},'
        '"ref_event_id":null,"source_span":"2 is a wizard",'
        '"parser_confidence":0.8}]}'
    )
    result = SpeechEventParser(SequenceBackend([response, response]), "parser").parse(
        utterance="2 is a wizard", utterance_id="u2", day=1,
        phase="speech", turn=5, speaker=1
    )
    assert result.status == "failed"
    assert result.events == ()

    camp_alias = (
        '{"events":[{"event_family":"BELIEF_ASSERTION","target":[2],'
        '"content":{"kind":"CAMP","value":"good"},"qualifier":{},'
        '"ref_event_id":null,"source_span":"2 is good",'
        '"parser_confidence":0.8}]}'
    )
    alias_result = SpeechEventParser(
        SequenceBackend([camp_alias, camp_alias]), "parser"
    ).parse(
        utterance="2 is good", utterance_id="u2b", day=1,
        phase="speech", turn=5, speaker=1
    )
    assert alias_result.status == "failed"
    assert alias_result.events == ()


def test_speech_parser_keeps_multi_event_anchor_without_duplicate_values():
    response = (
        '{"events":['
        '{"event_family":"BELIEF_ASSERTION","target":[2],'
        '"content":{"kind":"ROLE","value":"Seer"},'
        '"qualifier":{"certainty":"normal"},"ref_event_id":null,'
        '"source_span":"2 is Seer","parser_confidence":0.9},'
        '{"event_family":"SOCIAL_STANCE","target":[2],'
        '"content":{"kind":"STANCE","value":null},'
        '"qualifier":{"polarity":"negative","strength":"strong"},'
        '"ref_event_id":null,"source_span":"I distrust 2",'
        '"parser_confidence":0.8},'
        '{"event_family":"CLAIM_RESPONSE","target":[2],'
        '"content":{"kind":"RELATION","value":null},'
        '"qualifier":{"relation":"challenge"},"ref_event_id":null,'
        '"source_span":"I challenge that","parser_confidence":0.7}'
        ']}'
    )
    utterance = "2 is Seer; I distrust 2; I challenge that"
    result = SpeechEventParser(SequenceBackend([response]), "parser").parse(
        utterance=utterance, utterance_id="u3", day=1,
        phase="speech", turn=6, speaker=1
    )
    assert result.status == "ok"
    assert len(result.events) == 3
    assert {event["utterance_id"] for event in result.events} == {"u3"}
    assert result.events[1]["content"]["value"] is None
    assert result.events[1]["qualifier"]["polarity"] == "negative"
    assert result.events[2]["content"]["value"] is None
    assert result.events[2]["qualifier"]["relation"] == "challenge"
    assert all(event["source_span"] in utterance for event in result.events)
