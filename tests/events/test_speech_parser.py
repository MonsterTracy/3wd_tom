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
