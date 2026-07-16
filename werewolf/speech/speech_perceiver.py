import json
import re


class SpeechPerceiver:
    ALLOWED_PREDICATES = {
        "claim_role",
        "claim_camp",
        "counter_claim",
        "report_check_result",
        "suspect",
        "accuse_as_werewolf",
        "support",
        "oppose",
        "defend_self",
        "defend_other",
        "attack_logic",
        "question",
        "vote_intention",
        "follow_vote",
        "hedge",
        "retract",
    }
    ALLOWED_ROLES = {
        "Werewolf",
        "Seer",
        "Witch",
        "Guard",
        "Villager",
        "Unknown",
    }
    ALLOWED_POLARITIES = {"positive", "negative", "neutral"}
    ALLOWED_CERTAINTIES = {"explicit", "implicit", "hedge"}
    ALLOWED_CAMPS = {"Village", "Werewolf"}

    def __init__(self, backend=None, model_name=None):
        self.backend = backend
        self.model_name = model_name

    def parse(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
        context: dict | None = None,
    ) -> list[dict]:
        if self.backend is None or not self.model_name:
            return []

        try:
            prompt = self._build_prompt(speaker, speech, day, phase)
            response_text = self.backend.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0,
            )
            parsed = self._extract_json_array(response_text)
            return self._normalize(parsed, speaker, speech)
        except Exception:
            return []

    @staticmethod
    def _build_prompt(speaker: int, speech: str, day: int, phase: str) -> str:
        return f"""你是狼人杀发言结构化解析器。
你只抽取玩家在发言中表达了什么，不判断真假，不根据隐藏身份推理。

当前玩家：player{speaker}
当前天数：Day {day}
当前阶段：{phase}
玩家编号：1 到 7

允许的 predicate 只能是：
claim_role
claim_camp
counter_claim
report_check_result
suspect
accuse_as_werewolf
support
oppose
defend_self
defend_other
attack_logic
question
vote_intention
follow_vote
hedge
retract

允许的 role 只能是：
Werewolf
Seer
Witch
Guard
Villager
Unknown
null

允许的 camp 只能是：
Village
Werewolf
null

输出 JSON 数组。不要输出解释文字。

每个元素格式：
{{
  "speaker": int,
  "predicate": string,
  "target": int|null,
  "role": string|null,
  "camp": "Village"|"Werewolf"|null,
  "polarity": "positive"|"negative"|"neutral"|null,
  "certainty": "explicit"|"implicit"|"hedge",
  "condition": string|null,
  "source_text": string
}}

规则：
- speaker 必须等于当前发言玩家。
- target 只能是 1 到 7 的整数或 null。
- “我是好人”“我是好人阵营”“我是站好人边的”表示阵营声明，解析为 claim_camp，target 等于 speaker，role 为 null，camp 为 "Village"；不要解析为 role="Villager"。
- “我是狼人阵营”“我是狼队的”表示阵营声明，解析为 claim_camp，target 等于 speaker，role 为 null，camp 为 "Werewolf"。
- 只有“我是村民 / 我是平民 / 我是普通村民 / 我是预言家 / 我是女巫 / 我是守卫”等明确身份声明才使用 claim_role 和 role 字段。
- report_check_result 只用于发言者自己声称“我查验了X / 我验了X / 我摸了X”的情况。
- 如果只是引用别人查验结果，例如“6号给我发金水”“2号给3号金水”，不要解析成 report_check_result；这类引用应根据语义解析为 support / oppose / suspect / accuse_as_werewolf。
- “跟着X归票 / 听X归票 / 跟X投 / 跟X思路走”解析为 follow_vote，target 是被跟随的人。
- “投X / 出X / 票X / 归票X / 今天先投X”解析为 vote_intention，target 是被投票对象。
- 不要补充玩家没有说出的内容。
- 不要判断发言真假。
- 如果没有可抽取信息，输出 []。
- 只输出 JSON。

玩家发言：
{speech}"""

    @staticmethod
    def _extract_json_array(response_text: str) -> list:
        if not isinstance(response_text, str):
            raise ValueError("LLM response content must be text.")

        text = response_text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed

        raise ValueError("No JSON array found in LLM response.")

    @classmethod
    def _normalize(cls, parsed: list, speaker: int, speech: str) -> list[dict]:
        claims = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            predicate = item.get("predicate")
            if predicate not in cls.ALLOWED_PREDICATES:
                continue

            target = item.get("target")
            if type(target) is not int or not 1 <= target <= 7:
                target = None
            if predicate == "claim_role":
                target = speaker
            if predicate == "claim_camp":
                target = speaker

            role = item.get("role")
            if role not in cls.ALLOWED_ROLES:
                role = None
            if predicate == "accuse_as_werewolf":
                role = "Werewolf"
            if predicate == "claim_camp":
                role = None

            camp = item.get("camp")
            if camp not in cls.ALLOWED_CAMPS:
                camp = None

            polarity = item.get("polarity")
            if polarity not in cls.ALLOWED_POLARITIES:
                polarity = None

            certainty = item.get("certainty")
            if certainty not in cls.ALLOWED_CERTAINTIES:
                certainty = "implicit"

            condition = item.get("condition")
            if not isinstance(condition, str):
                condition = None

            source_text = item.get("source_text")
            if not isinstance(source_text, str) or not source_text.strip():
                source_text = speech

            claims.append(
                {
                    "speaker": speaker,
                    "predicate": predicate,
                    "target": target,
                    "role": role,
                    "camp": camp,
                    "polarity": polarity,
                    "certainty": certainty,
                    "condition": condition,
                    "source_text": source_text,
                }
            )
        return claims
