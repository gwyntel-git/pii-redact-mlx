import re
import json
import os
import difflib
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from tqdm import tqdm
from .faker_utils import FakePIIGenerator
from .backends import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OMLX_BASE_URL,
    DEFAULT_OMLX_CONCURRENCY,
    MAX_AUTO_NEW_TOKENS,
    BackendError,
    OMLXBackend,
    TransformersBackend,
    omlx_server_available,
)


class PIIHandlingMode(Enum):
    """Enum for different PII handling modes"""

    TAG = "tag"  # Keep PII content between XML tags: <PII:type>content</PII:type>
    REDACT = "redact"  # Replace PII with just an empty tag: <PII:type/>
    REPLACE = "replace"  # Replace PII values with fake data (tags are dropped)


class PIIType(Enum):
    """Enum for different PII types that can be identified and redacted"""

    AGE = "age"  # A person's age
    CREDIT_CARD_INFO = (
        "credit_card_info"  # A credit card number, expiration date, CCV, etc.
    )
    NATIONALITY = "nationality"  # A country when used to reference place of birth, residence, or citizenship
    DATE = "date"  # A specific calendar date
    DATE_OF_BIRTH = "date_of_birth"  # A specific calendar date representing birth
    DOMAIN_NAME = "domain_name"  # A domain on the internet
    EMAIL_ADDRESS = "email_address"  # An email ID
    DEMOGRAPHIC_GROUP = (
        "demographic_group"  # Anything that identifies race or ethnicity
    )
    GENDER = "gender"  # A gender identifier
    PERSONAL_ID = (
        "personal_id"  # Any ID string like a national ID, subscriber number, etc.
    )
    OTHER_ID = "other_id"  # Any ID not associated with a person like an organization ID, database ID, etc.
    BANKING_NUMBER = "banking_number"  # A number associated with a bank account
    MEDICAL_CONDITION = "medical_condition"  # A diagnosis, treatment code or other information identifying a medical condition
    ORGANIZATION_NAME = "organization_name"  # Name of an organization
    PERSON_NAME = "person_name"  # Name of a person
    PHONE_NUMBER = "phone_number"  # A telephone number
    STREET_ADDRESS = "street_address"  # A physical address
    PASSWORD = "password"  # A secure string used for authentication
    SECURE_CREDENTIAL = "secure_credential"  # Any secure credential like an API key, private key, 2FA token
    RELIGIOUS_AFFILIATION = (
        "religious_affiliation"  # Anything that identifies religious affiliation
    )


def parse_tagged_string(tagged_str):
    """
    Parses a tagged string (with PII tags) and returns a tuple (clean_str, annotations) where:
      - clean_str is the string with all tags removed.
      - annotations is a list of tuples (start, end, tag, annotated_text) for each annotated span.
    """
    annotations = []
    clean_str = ""
    i = 0
    clean_index = 0
    open_tag_pattern = re.compile(r"<PII:(\w+)>")

    while i < len(tagged_str):
        if tagged_str[i] == "<":
            m = open_tag_pattern.match(tagged_str, i)
            if m:
                tag = m.group(1)
                annotation_start = clean_index
                i = m.end()
                closing_tag = f"</PII:{tag}>"
                closing_index = tagged_str.find(closing_tag, i)

                if closing_index == -1:
                    clean_str += tagged_str[i]
                    clean_index += 1
                    i += 1
                    continue

                annotated_text = tagged_str[i:closing_index]
                annotations.append(
                    (
                        annotation_start,
                        annotation_start + len(annotated_text),
                        tag,
                        annotated_text,
                    )
                )
                clean_str += annotated_text
                clean_index += len(annotated_text)
                i = closing_index + len(closing_tag)
            else:
                clean_str += tagged_str[i]
                clean_index += 1
                i += 1
        else:
            clean_str += tagged_str[i]
            clean_index += 1
            i += 1
    return clean_str, annotations


def find_best_match(sub, original, start_hint, window=50):
    search_start = max(0, start_hint - window)
    pos = original.find(sub, search_start)
    if pos != -1:
        return pos

    best_ratio = 0.0
    best_index = -1
    search_end = min(len(original) - len(sub) + 1, start_hint + window)
    for i in range(search_start, search_end):
        candidate = original[i : i + len(sub)]
        ratio = difflib.SequenceMatcher(None, sub, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = i
    if best_ratio < 0.6:
        return -1
    return best_index


def merge_overlapping_spans(annotations):
    if not annotations:
        return []

    annotations.sort(key=lambda x: x[0])

    merged = []

    group_start, group_end, group_tag = annotations[0]
    best_length = group_end - group_start

    for ann in annotations[1:]:
        start, end, tag = ann

        if start <= group_end:
            group_end = max(group_end, end)
            length = end - start
            if length > best_length:
                best_length = length
                group_tag = tag
        else:
            merged.append((group_start, group_end, group_tag))
            group_start, group_end, group_tag = start, end, tag
            best_length = end - start

    merged.append((group_start, group_end, group_tag))
    return merged


def apply_tags(
    original, tagged_strings, tags_to_include, mode=PIIHandlingMode.TAG, locale="en_US"
):
    candidate_annotations = []

    for tstr, include_tags in zip(tagged_strings, tags_to_include):
        cleaned, annotations = parse_tagged_string(tstr)
        for ann_start, ann_end, tag, text in annotations:
            if include_tags != None and tag not in include_tags:
                continue

            rel = ann_start / len(cleaned) if cleaned else 0
            start_hint = int(rel * len(original))
            orig_start = find_best_match(text, original, start_hint)
            if orig_start == -1:
                continue
            orig_end = orig_start + len(text)
            candidate_annotations.append((orig_start, orig_end, tag, text))

    if mode == PIIHandlingMode.REPLACE:
        fake_generator = FakePIIGenerator(locale=locale)

    merge_input = [(start, end, tag) for start, end, tag, _ in candidate_annotations]

    merged_annotations = merge_overlapping_spans(merge_input)

    merged_with_text = []
    for start, end, tag in merged_annotations:
        original_text = original[start:end]
        merged_with_text.append((start, end, tag, original_text))

    inserts = {}

    for start, end, tag, text in merged_with_text:
        if mode == PIIHandlingMode.TAG:
            inserts[start] = f"<PII:{tag}>{text}</PII:{tag}>"
            for i in range(start + 1, end):
                inserts[i] = ""
        elif mode == PIIHandlingMode.REDACT:
            inserts[start] = f"<PII:{tag}/>"
            for i in range(start + 1, end):
                inserts[i] = ""
        elif mode == PIIHandlingMode.REPLACE:
            fake_value = fake_generator.get_fake_value(tag, text)
            inserts[start] = f"{fake_value}"
            for i in range(start + 1, end):
                inserts[i] = ""

    result = []
    i = 0
    while i <= len(original):
        if i in inserts:
            result.append(inserts[i])
        elif i < len(original):
            result.append(original[i])
        i += 1

    return "".join(result)


class PIIRedactor:
    def __init__(
        self,
        device=None,
        backend=None,
        model_paths=None,
        focus_tags=None,
        omlx_base_url=None,
        omlx_models=None,
        omlx_api_key=None,
        concurrency=None,
        max_tokens=None,
        auto_max_tokens=False,
    ):
        """
        Initialize the PIIRedactor with models for PII detection.

        Args:
            device (str): Device for the 'transformers' backend (e.g. 'cuda', 'cpu', 'mps').
            backend (str): Inference backend -- 'auto' (default), 'transformers', or 'omlx'.
                'auto' uses a reachable oMLX server when one is found, otherwise falls
                back to the local 'transformers' models. Also read from the
                PII_REDACT_BACKEND environment variable.
            model_paths (list): HuggingFace repo ids/paths used by the 'transformers' backend.
            focus_tags (list): Per-model tag allow-lists (None = keep every tag).
            omlx_base_url (str): Base URL of the oMLX OpenAI-compatible server,
                e.g. 'http://localhost:8000/v1'. Falls back to the OMLX_BASE_URL
                environment variable, then 'http://localhost:8000/v1'.
            omlx_models (list): oMLX model names, one per entry in model_paths.
            omlx_api_key (str): Optional bearer token for the oMLX server. Falls back
                to the OMLX_API_KEY environment variable.
            concurrency (int): Number of inference requests to run in parallel.
                ``None`` (default) resolves to ``DEFAULT_OMLX_CONCURRENCY`` (8) for
                the oMLX backend and 1 for the transformers backend. Also read from
                the PII_REDACT_CONCURRENCY environment variable. Model calls for all
                (document, model) pairs are dispatched to a thread pool, so both
                many documents and the two models per document run concurrently.
            max_tokens (int): Fixed generation budget per request. ``None`` (default)
                uses the backend default (1024). Ignored when ``auto_max_tokens`` is
                True. Also read from the PII_REDACT_MAX_TOKENS environment variable.
            auto_max_tokens (bool): Auto-size the generation budget per request from
                the input length. The redaction models echo their input back with
                tags, so the output is roughly as long as the input; the fixed 1024
                default truncates anything longer. Use this for long documents.
        """

        if max_tokens is None:
            env = os.environ.get("PII_REDACT_MAX_TOKENS")
            if env not in (None, ""):
                max_tokens = int(env)

        self.model_paths = model_paths or [
            "OpenPipe/Pii-Redact-Name",
            "OpenPipe/Pii-Redact-General",
        ]
        self.omlx_models = omlx_models or [
            "PII-Redact-Name",
            "PII-Redact-General",
        ]
        self.focus_tags = focus_tags or (
            [["person_name", "organization_name"]] + [None] * (len(self.model_paths) - 1)
        )

        self.device = device
        self.omlx_base_url = (
            omlx_base_url or os.environ.get("OMLX_BASE_URL") or DEFAULT_OMLX_BASE_URL
        )
        self.omlx_api_key = omlx_api_key or os.environ.get("OMLX_API_KEY")
        self.requested_backend = (
            backend or os.environ.get("PII_REDACT_BACKEND") or "auto"
        )
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.auto_max_tokens = auto_max_tokens

        # Resolved on first model call: 'auto' -> 'omlx' or 'transformers'.
        self.backend = None

        # Lazily-initialized inference backends, one per model role.
        self.models: list = [None] * len(self.model_paths)
        # Guards one-time lazy initialization across concurrent callers.
        self._init_lock = threading.Lock()

    def _resolve_backend(self):
        """Resolve the 'auto' backend choice to a concrete backend name."""
        if self.requested_backend in (None, "", "auto"):
            return "omlx" if omlx_server_available(self.omlx_base_url) else "transformers"
        return self.requested_backend

    def _resolve_concurrency(self):
        """Resolve the effective concurrency (falls back per backend)."""
        value = self.concurrency
        if value is None:
            env = os.environ.get("PII_REDACT_CONCURRENCY")
            if env not in (None, ""):
                value = int(env)
        if value is None:
            backend = self.backend or self._resolve_backend()
            value = DEFAULT_OMLX_CONCURRENCY if backend == "omlx" else 1
        return max(1, int(value))

    def _initialize_model(self, index):
        """Initialize a specific model backend if it hasn't been already."""
        if self.models[index] is not None:
            return

        with self._init_lock:
            if self.models[index] is not None:
                return

            if self.backend is None:
                self.backend = self._resolve_backend()

            if self.backend == "omlx":
                self.models[index] = OMLXBackend(
                    model=self.omlx_models[index],
                    base_url=self.omlx_base_url,
                    api_key=self.omlx_api_key,
                    concurrency=self._resolve_concurrency(),
                    max_tokens=None if self.auto_max_tokens else self.max_tokens,
                )
            elif self.backend == "transformers":
                self.models[index] = TransformersBackend(
                    self.model_paths[index],
                    device=self.device,
                    max_new_tokens=(
                        MAX_AUTO_NEW_TOKENS if self.auto_max_tokens else (self.max_tokens or DEFAULT_MAX_NEW_TOKENS)
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown backend {self.backend!r}: choose 'auto', 'transformers' or 'omlx'."
                )

    def _model_call(self, text, model_index):
        """
        Process text through a specific model to identify PII entities.

        Args:
            text (str): The text to process
            model_index (int): Index of the model to use

        Returns:
            str: The processed text with PII tags
        """
        self._initialize_model(model_index)
        return self.models[model_index].tag(text)

    def tag_pii_in_documents(
        self, documents, mode=PIIHandlingMode.TAG, locale="en_US", progress=False
    ):
        """
        Process a list of documents to identify and handle PII according to the specified mode.

        Inference calls for every (document, model) pair are dispatched to a thread
        pool sized by the resolved concurrency, so many documents -- and the two
        models per document -- run in parallel. Results are reassembled in document
        order, so output is identical to the sequential path.

        Args:
            documents (list): List of text documents to process.
            mode (PIIHandlingMode): How to handle identified PII:
                - TAG: Keep PII with XML tags
                - REDACT: Replace PII with empty tags
                - REPLACE: Replace PII with fake data
            locale (str): Locale for generating fake data (only used if mode=REPLACE)

        Returns:
            list: List of documents with PII handled according to the specified mode.
        """
        documents = list(documents)
        if not documents:
            return []

        num_models = len(self.model_paths)
        tasks = [
            (doc_index, model_index)
            for doc_index in range(len(documents))
            for model_index in range(num_models)
        ]

        concurrency = self._resolve_concurrency()
        outputs = {}

        if concurrency > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_task = {
                    executor.submit(self._model_call, documents[di], mi): (di, mi)
                    for di, mi in tasks
                }
                pending = as_completed(future_to_task)
                if progress:
                    pending = tqdm(
                        pending, total=len(tasks), desc="tagging", unit="req"
                    )
                for future in pending:
                    outputs[future_to_task[future]] = future.result()
        else:
            iterator = tasks
            if progress:
                iterator = tqdm(iterator, total=len(tasks), desc="tagging", unit="req")
            for di, mi in iterator:
                outputs[(di, mi)] = self._model_call(documents[di], mi)

        processed_documents = []
        for doc_index in range(len(documents)):
            model_outputs = [outputs[(doc_index, mi)] for mi in range(num_models)]
            processed_doc = apply_tags(
                documents[doc_index],
                model_outputs,
                self.focus_tags,
                mode=mode,
                locale=locale,
            )
            processed_documents.append(processed_doc)

        return processed_documents


def tag_pii_in_documents(
    documents,
    device=None,
    mode=PIIHandlingMode.TAG,
    locale="en_US",
    backend=None,
    omlx_base_url=None,
    omlx_models=None,
    omlx_api_key=None,
    concurrency=None,
    max_tokens=None,
    auto_max_tokens=False,
    progress=False,
):
    """
    Convenience function to process a list of documents through a PII tagging model.

    Args:
        documents (list): List of text documents to process.
        device (str): Device for the 'transformers' backend (e.g. 'cuda', 'cpu').
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)
        backend (str): 'auto' (default), 'transformers', or 'omlx'. 'auto' uses a
            reachable oMLX server if one is found, else local transformers models.
        omlx_base_url (str): Base URL of the oMLX OpenAI-compatible server.
        omlx_models (list): oMLX model names (name model, general model).
        omlx_api_key (str): Optional bearer token for the oMLX server.
        concurrency (int): Number of inference requests to run in parallel. None
            (default) resolves to 8 for oMLX and 1 for transformers.

    Returns:
        list: List of documents with PII handled according to the specified mode.
    """
    redactor = PIIRedactor(
        device=device,
        backend=backend,
        omlx_base_url=omlx_base_url,
        omlx_models=omlx_models,
        omlx_api_key=omlx_api_key,
        concurrency=concurrency,
        max_tokens=max_tokens,
        auto_max_tokens=auto_max_tokens,
    )
    return redactor.tag_pii_in_documents(
        documents, mode=mode, locale=locale, progress=progress
    )


def clean_dataset(
    input_filename,
    output_filename,
    device=None,
    mode=PIIHandlingMode.TAG,
    locale="en_US",
    backend=None,
    omlx_base_url=None,
    omlx_models=None,
    omlx_api_key=None,
    concurrency=None,
    batch_size=None,
    max_tokens=None,
    auto_max_tokens=False,
):
    """
    Reads a JSONL dataset and processes the 'content' field in each message.
    Processes JSON objects, updates them with the processed messages,
    and writes them immediately to the output file. This allows progress to be saved incrementally.

    Lines are read in batches and every message in a batch is tagged in one
    concurrent call, so the thread pool is kept busy across documents while
    output is still flushed after each batch.

    Args:
        input_filename (str): Path to the input JSONL file.
        output_filename (str): Path to the output JSONL file.
        device (str): Device for the 'transformers' backend (e.g. 'cuda', 'cpu').
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)
        backend (str): 'auto' (default), 'transformers', or 'omlx'. 'auto' uses a
            reachable oMLX server if one is found, else local transformers models.
        omlx_base_url (str): Base URL of the oMLX OpenAI-compatible server.
        omlx_models (list): oMLX model names (name model, general model).
        omlx_api_key (str): Optional bearer token for the oMLX server.
        concurrency (int): Number of parallel inference requests. None (default)
            resolves to 8 for oMLX and 1 for transformers.
        batch_size (int): Number of JSONL lines processed per concurrent call.
            None (default) uses ``max(16, concurrency * 4)``.
    """
    redactor = PIIRedactor(
        device=device,
        backend=backend,
        omlx_base_url=omlx_base_url,
        omlx_models=omlx_models,
        omlx_api_key=omlx_api_key,
        concurrency=concurrency,
        max_tokens=max_tokens,
        auto_max_tokens=auto_max_tokens,
    )

    if batch_size is None:
        batch_size = max(16, redactor._resolve_concurrency() * 4)
    batch_size = max(1, int(batch_size))

    with open(input_filename, "r") as f:
        num_lines = sum(1 for line in f)

    with open(input_filename, "r") as fin, open(output_filename, "w") as fout:
        batch = []
        for line in tqdm(fin, total=num_lines):
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                process_and_write_batch(
                    batch, fout, redactor, mode=mode, locale=locale
                )
                batch = []
        if batch:
            process_and_write_batch(batch, fout, redactor, mode=mode, locale=locale)


def process_and_write_batch(
    json_objs_batch, fout, redactor, mode=PIIHandlingMode.TAG, locale="en_US"
):
    """
    Given a batch of JSON objects, extracts all messages, processes them,
    updates the JSON objects, and writes them to the provided output file.

    Args:
        json_objs_batch (list): List of JSON objects.
        fout (file object): Open output file to write processed JSON objects.
        redactor (PIIRedactor): Redactor object to use for tagging.
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)
    """
    messages_to_process = []
    for obj in json_objs_batch:
        for message in obj.get("messages", []):
            if message["content"]:
                messages_to_process.append(message["content"])

    processed_messages = redactor.tag_pii_in_documents(
        messages_to_process, mode=mode, locale=locale
    )

    msg_idx = 0
    for obj in json_objs_batch:
        if "messages" in obj:
            for i in range(len(obj["messages"])):
                if obj["messages"][i]["content"]:
                    obj["messages"][i]["content"] = processed_messages[msg_idx]
                    msg_idx += 1

        fout.write(json.dumps(obj) + "\n")
    fout.flush()
