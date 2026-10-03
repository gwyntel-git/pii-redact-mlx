#!/usr/bin/env python
import argparse
import os
import sys
from .redactor import clean_dataset, tag_pii_in_documents, PIIHandlingMode
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


def _omlx_models(args):
    return [args.omlx_model_name, args.omlx_model_general]


def main():
    parser = argparse.ArgumentParser(description="PII Redaction Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add common arguments for PII handling
    def add_pii_handling_args(parser):
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
        parser.set_defaults(mode=PIIHandlingMode.TAG)

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
