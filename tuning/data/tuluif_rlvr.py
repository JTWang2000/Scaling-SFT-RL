import json

from datasets import Dataset, DatasetDict

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING


TULUIF_FUNC_TO_INSTRUCTION = {
    "validate_choice": "detectable_format:constrained_response",
    "validate_end": "startend:end_checker",
    "validate_forbidden_words": "keywords:forbidden_words",
    "validate_frequency_capital_words": "change_case:capital_word_frequency",
    "validate_highlighted_sections": "detectable_format:number_highlighted_sections",
    "validate_json_format": "detectable_format:json_format",
    "validate_lowercase": "change_case:english_lowercase",
    "validate_no_commas": "punctuation:no_comma",
    "validate_paragraphs": "length_constraints:nth_paragraph_first_word",
    "validate_placeholders": "detectable_content:number_placeholders",
    "validate_quotation": "startend:quotation",
    "validate_repeat_prompt": "combination:repeat_prompt",
    "validate_sections": "detectable_format:multiple_sections",
    "validate_title": "detectable_format:title",
    "validate_two_responses": "combination:two_responses",
    "validate_uppercase": "change_case:english_capital",
    "validate_word_constraint": "length_constraints:number_words",
    "verify_bullet_points": "detectable_format:number_bullet_lists",
    "verify_keyword_frequency": "keywords:frequency",
    "verify_keywords": "keywords:existence",
    "verify_letter_frequency": "keywords:letter_frequency",
    "verify_paragraph_count": "length_constraints:number_paragraphs",
    "verify_postscript": "detectable_content:postscript",
    "verify_sentence_constraint": "length_constraints:number_sentences",
}


class TuluIFRLVR(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="tuluif")

    def _extract_prompt(self, example):
        messages = example.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content

        return example.get("prompt") or example.get("instruction") or example.get("input") or ""

    def _extract_instruction_spec(self, example):
        raw = example.get("ground_truth")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}

        func_name = raw.get("func_name")
        if not func_name:
            return [], []
        return self._map_instruction_spec(func_name, raw)

    def _relation_spec(self, quantifier, value_key, value):
        if value is None:
            return [], []

        if quantifier == "at most":
            return [value_key], [{"relation": "less than", value_key: int(value) + 1}]
        if quantifier == "around":
            return (
                [value_key, value_key],
                [
                    {"relation": "at least", value_key: max(0, int(value) - 1)},
                    {"relation": "less than", value_key: int(value) + 2},
                ],
            )
        return [value_key], [{"relation": "at least", value_key: int(value)}]

    def _map_instruction_spec(self, func_name, raw):
        instruction_id = TULUIF_FUNC_TO_INSTRUCTION.get(func_name)
        if instruction_id is None:
            return [], []

        quantifier = raw.get("quantifier")
        if func_name == "validate_end":
            return [instruction_id], [{"end_phrase": raw.get("end_phrase")}]
        if func_name == "validate_forbidden_words":
            return [instruction_id], [{"forbidden_words": raw.get("forbidden_words") or []}]
        if func_name == "validate_frequency_capital_words":
            keys, kwargs = self._relation_spec(quantifier, "capital_frequency", raw.get("N"))
            return [instruction_id] * len(keys), [
                {**item, "capital_relation": item.pop("relation")} for item in kwargs
            ]
        if func_name == "validate_highlighted_sections":
            return [instruction_id], [{"num_highlights": raw.get("N")}]
        if func_name == "validate_paragraphs":
            return [instruction_id], [
                {
                    "num_paragraphs": raw.get("N"),
                    "nth_paragraph": raw.get("i"),
                    "first_word": raw.get("first_word"),
                }
            ]
        if func_name == "validate_placeholders":
            return [instruction_id], [{"num_placeholders": raw.get("N")}]
        if func_name == "validate_repeat_prompt":
            return [instruction_id], [{"prompt_to_repeat": raw.get("original_prompt")}]
        if func_name == "validate_sections":
            return [instruction_id], [
                {"section_spliter": raw.get("section_splitter"), "num_sections": raw.get("N")}
            ]
        if func_name == "validate_word_constraint":
            keys, kwargs = self._relation_spec(quantifier, "num_words", raw.get("N"))
            return [instruction_id] * len(keys), kwargs
        if func_name == "verify_bullet_points":
            return [instruction_id], [{"num_bullets": raw.get("N")}]
        if func_name == "verify_keyword_frequency":
            keys, kwargs = self._relation_spec(quantifier or "at least", "frequency", raw.get("N"))
            return [instruction_id] * len(keys), [
                {**item, "keyword": raw.get("word")} for item in kwargs
            ]
        if func_name == "verify_keywords":
            return [instruction_id], [{"keywords": raw.get("keyword_list") or []}]
        if func_name == "verify_letter_frequency":
            keys, kwargs = self._relation_spec(quantifier or "at least", "let_frequency", raw.get("N"))
            return [instruction_id] * len(keys), [
                {**item, "letter": raw.get("letter"), "let_relation": item.pop("relation")} for item in kwargs
            ]
        if func_name == "verify_paragraph_count":
            return [instruction_id], [{"num_paragraphs": raw.get("N")}]
        if func_name == "verify_postscript":
            return [instruction_id], [{"postscript_marker": raw.get("postscript_marker")}]
        if func_name == "verify_sentence_constraint":
            keys, kwargs = self._relation_spec(quantifier, "num_sentences", raw.get("N"))
            return [instruction_id] * len(keys), kwargs

        return [instruction_id], [{}]

    def _normalize_row(self, example):
        prompt = self._extract_prompt(example)
        instruction_id_list = example.get("instruction_id_list") or example.get("instruction_ids")
        kwargs = example.get("kwargs")

        if instruction_id_list is None and kwargs is None:
            instruction_id_list, kwargs = self._extract_instruction_spec(example)
        else:
            instruction_id_list = instruction_id_list or []
            kwargs = kwargs or []

        if isinstance(instruction_id_list, str):
            instruction_id_list = [instruction_id_list]
        if not isinstance(instruction_id_list, list):
            instruction_id_list = []

        if isinstance(kwargs, dict):
            kwargs = [kwargs]
        if not isinstance(kwargs, list):
            kwargs = []

        if len(kwargs) < len(instruction_id_list):
            kwargs = kwargs + [{} for _ in range(len(instruction_id_list) - len(kwargs))]
        elif len(kwargs) > len(instruction_id_list):
            kwargs = kwargs[: len(instruction_id_list)]

        return {
            "system_message": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING,
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
        }

    def format_dataset(self):
        if isinstance(self._dataset, DatasetDict):
            if "train" in self._dataset:
                source = self._dataset["train"]
            else:
                source = self._dataset[next(iter(self._dataset.keys()))]
        else:
            source = self._dataset

        rows = [self._normalize_row(example) for example in source]
        rows = [row for row in rows if row["prompt"].strip()]

        if not rows:
            raise ValueError("No TuluIF RLVR rows produced after normalization. Check source schema.")

        formatted = Dataset.from_list(rows)
        test_size = 200 if len(formatted) > 200 else max(1, len(formatted) // 10)
        formatted_dataset = formatted.train_test_split(test_size=test_size, shuffle=False)

        print(f"Tuluif rlvr dataset - {formatted_dataset}")
        if len(formatted_dataset["train"]):
            print(f'Example Tuluif rlvr row - {formatted_dataset["train"][0]}')
        print("***")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    tuluif = TuluIFRLVR()

    tuluif.load_from_huggingface("allenai/RLVR-IFeval")
    tuluif.format_dataset()
    tuluif.clear_old_datasets(prefix="rlvr-tuluif")
    tuluif.save_dataset_to_disk(save_name="rlvr-tuluif")
