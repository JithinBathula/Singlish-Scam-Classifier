#!/usr/bin/env python3
"""Localize SMSSpamCollection train/test splits with OpenRouter.

This script reads the existing split files:
- datasets/SMSSpamCollection_train_subset
- datasets/SMSSpamCollection_ablation_test

It localizes each message into a Singapore-context SMS variant in batches of 50
and writes separate CSV outputs with:
- source_index
- raw_label
- label
- original_text
- localized_text
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-4.6"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_PAUSE_SECONDS = 1.0

LABEL_MAP = {
    "ham": "not_scam",
    "spam": "scam",
}

OUTPUT_FIELDNAMES = [
    "source_index",
    "raw_label",
    "label",
    "original_text",
    "localized_text",
]

SYSTEM_PROMPT = """You are helping build a Singapore-localized SMS scam classification dataset.

For each input message:
- Preserve the original meaning and intent.
- Preserve whether the message is scam/spam or not_scam/normal.
- Rewrite it so it sounds natural in a Singapore SMS context.
- Adapt references such as banks, payment methods, delivery services, agencies, and locations only when it feels natural.
- Keep the message concise and SMS-like.
- Make the rewrite visibly localized. Avoid outputs that only fix punctuation, casing, or one typo.
- For normal messages, make the tone clearly Singlish and conversational. Use natural particles and phrasing such as lah, lor, leh, hor, can, can or not, like that, one, sia when they fit.
- Most normal messages should feel obviously more Singaporean than the source text.
- For normal messages, prefer sounding like a real casual Singapore texter, even if that means adding a small amount of Singlish that was not explicit in the source.
- It is fine for many normal messages to include one or two Singlish markers so the localization is obvious.
- For scam messages, replace obviously foreign references with plausible Singapore-facing equivalents when appropriate.
- For scam messages, it is fine to keep the threatening or promotional tone, but make the message feel targeted at Singapore recipients.
- If the source is already somewhat local, keep that flavor and strengthen it slightly instead of leaving it unchanged.
- Do not add explanations, labels, numbering, or extra commentary.
- Use Singlish-like phrasing only when it makes sense, and do not overdo it.
- Do not leave the rewrite almost unchanged unless the source is already clearly Singaporean and Singlish in style.

Return valid JSON only.
"""

EXAMPLE_BLOCK = """Examples:

Original: "Busy here. Trying to finish for new year. I am looking forward to finally meeting you..."
Localized: "Busy here lah, trying to finish before CNY. Looking forward to finally meet you."

Original: "Todays Voda numbers ending 7548 are selected to receive a $350 award."
Localized: "Todays Singtel numbers ending 7548 are selected to receive a $350 award."

Original: "Ya tel, wats ur problem.."
Localized: "Ya tell me leh, what your problem sia.."

Original: "Good sleep is about rhythm. The person has to establish a rhythm that the body will learn and use."
Localized: "Good sleep is about rhythm one. Need to let your body get used to the pattern then can."

Original: "No break time one... How... I come out n get my stuff fr ü?"
Localized: "No break time leh... How like that... I come out take my stuff from you can or not?"

Original: "Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award!"
Localized: "Dear Dave this is your final notice to collect your 4* Bali holiday or $5000 CASH award!"
"""

LOCAL_MARKERS = {
    "singapore",
    "singaporean",
    "sg",
    "singtel",
    "starhub",
    "m1",
    "dbs",
    "ocbc",
    "uob",
    "paynow",
    "paylah",
    "singpost",
    "hdb",
    "mrt",
    "smrt",
    "nus",
    "ntu",
    "nyp",
    "jurong",
    "tampines",
    "bishan",
    "toa payoh",
    "woodlands",
    "ang mo kio",
    "bugis",
    "orchard",
    "cny",
    "cpf",
    "ica",
    "mom",
    "lah",
    "lor",
    "leh",
    "liao",
    "sia",
    "wah",
    "alamak",
    "paiseh",
    "shiok",
    "can or not",
    "makan",
    "chope",
}

CASUAL_SINGLISH_PATTERNS = [
    r"\blah\b",
    r"\blor\b",
    r"\bleh\b",
    r"\bhor\b",
    r"\bmeh\b",
    r"\blia[o]?\b",
    r"\bsia\b",
    r"\bah\b",
    r"\bcan or not\b",
    r"\blike that\b",
    r"\bpaiseh\b",
    r"\balamak\b",
    r"\bshiok\b",
    r"\bmakan\b",
    r"\bchope\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize SMSSpamCollection split files with OpenRouter."
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to the .env file containing OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "SMSSpamCollection_train_subset",
        help="Input path for the train subset.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "SMSSpamCollection_ablation_test",
        help="Input path for the ablation test subset.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "SMSSpamCollection_train_subset_localized.csv",
        help="Output CSV path for the localized train subset.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "SMSSpamCollection_ablation_test_localized.csv",
        help="Output CSV path for the localized ablation test subset.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to localize per API call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per batch if the API call or JSON parsing fails.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout for each OpenRouter request.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Pause between successful batches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing localized CSVs and regenerate them from scratch.",
    )
    return parser.parse_args()


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_openrouter_api_key(env_path: Path) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    env_values = load_env_file(env_path)
    api_key = env_values.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    raise RuntimeError(
        "OPENROUTER_API_KEY was not found in the environment or the .env file."
    )


def load_sms_split(input_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw_index, raw_line in enumerate(fp):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(
                    f"{input_path} line {raw_index + 1} is missing a tab separator."
                )

            raw_label, original_text = line.split("\t", 1)
            raw_label = raw_label.strip()
            original_text = original_text.strip()
            if raw_label not in LABEL_MAP:
                raise ValueError(
                    f"{input_path} line {raw_index + 1} has unexpected label {raw_label!r}."
                )
            if not original_text:
                continue

            rows.append(
                {
                    "source_index": len(rows),
                    "raw_label": raw_label,
                    "label": LABEL_MAP[raw_label],
                    "original_text": original_text,
                }
            )
    return rows


def chunked(rows: list[dict[str, object]], batch_size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_user_prompt(batch_rows: list[dict[str, object]]) -> str:
    compact_rows = [
        {
            "source_index": row["source_index"],
            "label": row["label"],
            "original_text": row["original_text"],
        }
        for row in batch_rows
    ]
    return (
        "Localize each SMS message for a general Singapore context.\n"
        "Return a JSON array with exactly the same number of items and the same source_index values.\n"
        "Each array item must be an object with:\n"
        '- "source_index": integer\n'
        '- "localized_text": string\n'
        "Do not include any extra keys or any text outside the JSON array.\n\n"
        "Aim for clearly localized rewrites, not just proofreading.\n"
        "Most outputs should contain a visible Singapore-context adaptation in wording, register, or references.\n"
        "For normal messages, prefer noticeably Singlish wording instead of plain English paraphrases.\n"
        "For many normal messages, add 1 to 2 natural Singlish particles or phrases so the localization is obvious.\n"
        "If a message comes back almost unchanged, that is usually too weak.\n"
        "Keep labels intact and do not turn normal messages into scams or vice versa.\n\n"
        f"{EXAMPLE_BLOCK}\n\n"
        f"Input JSON:\n{json.dumps(compact_rows, ensure_ascii=False, indent=2)}"
    )


def extract_message_text(response_json: dict[str, object]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response format: {response_json}") from exc

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            return "".join(text_parts)

    raise RuntimeError(f"Could not extract text content from response: {response_json}")


def parse_json_array_from_text(raw_text: str) -> list[dict[str, object]]:
    candidates: list[str] = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and start < end:
        candidates.insert(0, raw_text[start : end + 1].strip())

    seen_candidates: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            for key in ("items", "results", "localizations"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break

        if isinstance(parsed, list):
            return parsed

    raise ValueError(f"Model output was not a valid JSON array.\nRaw output:\n{raw_text}")


def normalize_batch_results(
    batch_rows: list[dict[str, object]],
    model_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    result_map: dict[int, str] = {}

    for item in model_results:
        if not isinstance(item, dict):
            raise ValueError(f"Each output item must be an object. Got: {item!r}")

        if "source_index" not in item:
            raise ValueError(f"Missing source_index in output item: {item!r}")

        try:
            source_index = int(item["source_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source_index in output item: {item!r}") from exc

        localized_text = item.get("localized_text")
        if not isinstance(localized_text, str) or not localized_text.strip():
            raise ValueError(f"Missing or empty localized_text in output item: {item!r}")

        if source_index in result_map:
            raise ValueError(f"Duplicate source_index {source_index} in model output.")
        result_map[source_index] = localized_text.strip()

    expected_indices = {int(row["source_index"]) for row in batch_rows}
    returned_indices = set(result_map)
    if returned_indices != expected_indices:
        raise ValueError(
            f"Returned source_index values did not match batch.\n"
            f"Expected: {sorted(expected_indices)}\n"
            f"Returned: {sorted(returned_indices)}"
        )

    normalized_rows: list[dict[str, object]] = []
    for row in batch_rows:
        source_index = int(row["source_index"])
        normalized_rows.append(
            {
                "source_index": source_index,
                "raw_label": row["raw_label"],
                "label": row["label"],
                "original_text": row["original_text"],
                "localized_text": result_map[source_index],
            }
        )
    return normalized_rows


def normalize_for_identity(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_local_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LOCAL_MARKERS)


def has_casual_singlish_marker(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in CASUAL_SINGLISH_PATTERNS)


def is_weak_localization(label: str, original_text: str, localized_text: str) -> bool:
    original_normalized = normalize_for_identity(original_text)
    localized_normalized = normalize_for_identity(localized_text)

    if original_normalized == localized_normalized:
        return not has_local_marker(original_text)

    similarity = difflib.SequenceMatcher(
        None,
        collapse_whitespace(original_text),
        collapse_whitespace(localized_text),
    ).ratio()

    if similarity >= 0.985 and not has_local_marker(localized_text) and not has_local_marker(original_text):
        return True

    if label == "not_scam":
        if not has_casual_singlish_marker(localized_text):
            return True

    return False


def validate_localized_rows(rows: list[dict[str, object]]) -> list[int]:
    weak_indices: list[int] = []
    for row in rows:
        label = str(row["label"])
        original_text = str(row["original_text"])
        localized_text = str(row["localized_text"])
        if is_weak_localization(label, original_text, localized_text):
            weak_indices.append(int(row["source_index"]))
    return weak_indices


def call_openrouter(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 12000,
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "smsspam-localization",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed with status {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    response_json = json.loads(response_body)
    if "error" in response_json:
        raise RuntimeError(f"OpenRouter returned an error: {response_json['error']}")

    return extract_message_text(response_json)


def localize_batch(
    *,
    api_key: str,
    model: str,
    batch_rows: list[dict[str, object]],
    timeout_seconds: int,
    max_retries: int,
    split_name: str,
    batch_number: int,
    total_batches: int,
) -> list[dict[str, object]]:
    user_prompt = build_user_prompt(batch_rows)

    for attempt in range(1, max_retries + 1):
        try:
            response_text = call_openrouter(
                api_key=api_key,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )
            parsed_results = parse_json_array_from_text(response_text)
            normalized_rows = normalize_batch_results(batch_rows, parsed_results)
            return normalized_rows
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(
                    f"[{split_name}] batch {batch_number}/{total_batches} failed after "
                    f"{max_retries} attempts."
                ) from exc

            backoff_seconds = 2 ** attempt
            print(
                f"[{split_name}] batch {batch_number}/{total_batches} attempt "
                f"{attempt}/{max_retries} failed: {exc}\nRetrying in {backoff_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(backoff_seconds)

    raise AssertionError("Unreachable code path.")


def load_completed_source_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()

    completed_indices: set[int] = set()
    with output_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        missing_columns = set(OUTPUT_FIELDNAMES) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Existing output file {output_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            completed_indices.add(int(row["source_index"]))
    return completed_indices


def append_rows_to_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_output_csv(output_path: Path) -> None:
    if not output_path.exists():
        return

    with output_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    rows.sort(key=lambda row: int(row["source_index"]))

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_split(
    *,
    split_name: str,
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    batch_size: int,
    timeout_seconds: int,
    max_retries: int,
    pause_seconds: float,
    overwrite: bool,
) -> None:
    if overwrite and output_path.exists():
        output_path.unlink()

    rows = load_sms_split(input_path)
    completed_indices = load_completed_source_indices(output_path)
    pending_rows = [row for row in rows if int(row["source_index"]) not in completed_indices]

    print(
        f"[{split_name}] loaded {len(rows)} rows from {input_path}. "
        f"{len(completed_indices)} already localized, {len(pending_rows)} pending."
    )

    if not pending_rows:
        sort_output_csv(output_path)
        print(f"[{split_name}] nothing to do. Output already complete: {output_path}")
        return

    pending_batches = list(chunked(pending_rows, batch_size))
    total_batches = len(pending_batches)

    for batch_number, batch_rows in enumerate(pending_batches, start=1):
        localized_rows = localize_batch(
            api_key=api_key,
            model=model,
            batch_rows=batch_rows,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            split_name=split_name,
            batch_number=batch_number,
            total_batches=total_batches,
        )
        append_rows_to_csv(output_path, localized_rows)
        print(
            f"[{split_name}] finished batch {batch_number}/{total_batches} "
            f"({len(batch_rows)} rows)."
        )
        if pause_seconds > 0 and batch_number < total_batches:
            time.sleep(pause_seconds)

    sort_output_csv(output_path)
    print(f"[{split_name}] wrote localized CSV to {output_path}")


def main() -> int:
    args = parse_args()
    api_key = get_openrouter_api_key(args.env_path)

    process_split(
        split_name="train",
        input_path=args.train_input,
        output_path=args.train_output,
        api_key=api_key,
        model=args.model,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        pause_seconds=args.pause_seconds,
        overwrite=args.overwrite,
    )
    process_split(
        split_name="test",
        input_path=args.test_input,
        output_path=args.test_output,
        api_key=api_key,
        model=args.model,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        pause_seconds=args.pause_seconds,
        overwrite=args.overwrite,
    )

    print("Localization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
