#!/usr/bin/env python
import argparse
import json
import os
import sys
from .redactor import clean_dataset, tag_pii_in_documents, PIIHandlingMode
from .traces import convert_traces
from .backends import (
    DEFAULT_OMLX_BASE_URL,
    BackendError,
    list_omlx_models,
)


def add_backend_args(parser):
    """Add backend-selection arguments shared by the processing commands."""
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "omlx"],
        default=os.environ.get("PII_REDACT_BACKEND", "auto"),
        help="Inference backend. 'auto' (default) uses a reachable oMLX server when "
        "one is found, otherwise the local transformers models. "
        "(env: PII_REDACT_BACKEND)",
    )
    parser.add_argument(
        "--omlx-url",
        default=None,
        help="oMLX OpenAI-compatible base URL, e.g. http://localhost:8000/v1 "
        f"(default: $OMLX_BASE_URL or {DEFAULT_OMLX_BASE_URL})",
    )
    parser.add_argument(
        "--omlx-model-name",
        default="PII-Redact-Name",
        help="oMLX model name for the person/organization model "
        "(default: PII-Redact-Name)",
    )
    parser.add_argument(
        "--omlx-model-general",
        default="PII-Redact-General",
        help="oMLX model name for the general model (default: PII-Redact-General)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of inference requests to run in parallel (default: 8 for the "
        "omlx backend, 1 for transformers). (env: PII_REDACT_CONCURRENCY)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Fixed generation budget per request (default 1024). The redaction "
        "models echo the input back with tags, so long documents need a larger "
        "budget -- see --auto-max-tokens. (env: PII_REDACT_MAX_TOKENS)",
    )
    parser.add_argument(
        "--auto-max-tokens",
        action="store_true",
        help="Auto-size the generation budget from each input's length (use for "
        "documents longer than ~1024 tokens).",
    )


def _omlx_models(args):
    return [args.omlx_model_name, args.omlx_model_general]


def main():
    parser = argparse.ArgumentParser(description="PII Redaction Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add common arguments for PII handling
    def add_pii_handling_args(parser, default_mode=PIIHandlingMode.TAG):
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--tag",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.TAG,
            help="Keep PII content between XML tags (default)",
        )
        mode_group.add_argument(
            "--redact",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.REDACT,
            help="Replace PII with just an empty tag",
        )
        mode_group.add_argument(
            "--replace",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.REPLACE,
            help="Replace PII with fake data",
        )
        parser.set_defaults(mode=default_mode)

        parser.add_argument(
            "--locale",
            default="en_US",
            help="Locale for generating fake data (default: en_US, only used with --replace)",
        )

    # Process JSONL dataset command
    jsonl_parser = subparsers.add_parser(
        "process-jsonl", help="Process a JSONL dataset to handle PII in message content"
    )
    jsonl_parser.add_argument("input", help="Input JSONL file")
    jsonl_parser.add_argument("output", help="Output JSONL file")
    jsonl_parser.add_argument(
        "--device", help="Device to use for processing (e.g., cuda, cpu, mps)"
    )
    jsonl_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of JSONL lines processed per concurrent batch "
        "(default: max(16, concurrency * 4))",
    )
    add_pii_handling_args(jsonl_parser)
    add_backend_args(jsonl_parser)

    # Process text files command
    text_parser = subparsers.add_parser(
        "process-text", help="Process a text file with one document per line"
    )
    text_parser.add_argument("input", help="Input file with one document per line")
    text_parser.add_argument("output", help="Output file for processed documents")
    text_parser.add_argument(
        "--device", help="Device to use for processing (e.g., cuda, cpu, mps)"
    )
    add_pii_handling_args(text_parser)
    add_backend_args(text_parser)

    # Convert captured request traces into a redacted OpenAI messages JSONL
    traces_parser = subparsers.add_parser(
        "convert-traces",
        help="Redact a trace JSONL (captured chat requests) into an OpenAI messages JSONL",
    )
    traces_parser.add_argument("input", help="Input trace JSONL file")
    traces_parser.add_argument("output", help="Output OpenAI messages JSONL file")
    traces_parser.add_argument(
        "--request-field",
        default=None,
        help="Record field holding the chat request (default: auto-detect "
        "transformed_request/raw_request/request/body/payload)",
    )
    traces_parser.add_argument(
        "--keep-field",
        action="append",
        default=[],
        dest="keep_fields",
        metavar="FIELD",
        help="Copy this top-level record field into each output line (repeatable)",
    )
    traces_parser.add_argument(
        "--device", help="Device to use for the transformers backend"
    )
    add_pii_handling_args(traces_parser, default_mode=PIIHandlingMode.REDACT)
    add_backend_args(traces_parser)

    # List models served by an oMLX server
    list_parser = subparsers.add_parser(
        "list-models", help="List the models served by an oMLX server"
    )
    list_parser.add_argument(
        "--omlx-url",
        default=None,
        help="oMLX OpenAI-compatible base URL "
        f"(default: $OMLX_BASE_URL or {DEFAULT_OMLX_BASE_URL})",
    )

    args = parser.parse_args()

    try:
        if args.command == "process-jsonl":
            clean_dataset(
                args.input,
                args.output,
                device=args.device,
                mode=args.mode,
                locale=args.locale,
                backend=args.backend,
                omlx_base_url=args.omlx_url,
                omlx_models=_omlx_models(args),
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                auto_max_tokens=args.auto_max_tokens,
            )
        elif args.command == "process-text":
            with open(args.input, "r") as f:
                documents = [line.strip() for line in f if line.strip()]

            tagged_documents = tag_pii_in_documents(
                documents,
                device=args.device,
                mode=args.mode,
                locale=args.locale,
                backend=args.backend,
                omlx_base_url=args.omlx_url,
                omlx_models=_omlx_models(args),
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                auto_max_tokens=args.auto_max_tokens,
            )

            with open(args.output, "w") as f:
                for doc in tagged_documents:
                    f.write(doc + "\n")

            mode_descriptions = {
                PIIHandlingMode.TAG: "Tagged",
                PIIHandlingMode.REDACT: "Redacted",
                PIIHandlingMode.REPLACE: "Replaced",
            }
            action = mode_descriptions[args.mode]
            print(
                f"{action} PII in {len(tagged_documents)} documents and saved to {args.output}"
            )
        elif args.command == "convert-traces":
            stats = convert_traces(
                args.input,
                args.output,
                request_field=args.request_field,
                mode=args.mode,
                locale=args.locale,
                keep_fields=args.keep_fields,
                progress=True,
                backend=args.backend,
                omlx_base_url=args.omlx_url,
                omlx_models=_omlx_models(args),
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                auto_max_tokens=args.auto_max_tokens,
            )
            print(json.dumps(stats, indent=2))
        elif args.command == "list-models":
            for model in list_omlx_models(args.omlx_url):
                print(model)
        else:
            parser.print_help()
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
