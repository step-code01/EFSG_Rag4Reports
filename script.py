#!/usr/bin/env python3
"""
EFSG Submission for RAG4Reports ACL 2026
Adapted for TIRA via GitHub Actions
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Initialize models FIRST
from efsg_components import (
    initialize_models, TopicJSON, EFSGPipeline, LocalCorpus
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="efsg-submission")
    parser.add_argument("-i", "--input", required=True, help="Input directory with report-requests.jsonl")
    parser.add_argument("-o", "--output", required=True, help="Output directory for run.jsonl")
    return parser.parse_args()


def load_corpus():
    """Load RAGTIME1 corpus from HuggingFace."""
    try:
        from efsg_components import LocalCorpus
        print('[TIRA] Loading corpus from HuggingFace...')
        corpus = LocalCorpus(cache_size=50_000, language='eng')
        return corpus
    except Exception as e:
        print(f'[TIRA] Corpus load failed: {e}')
        raise


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize global models
    print('[TIRA] Initializing EFSG...')
    initialize_models()

    # Load corpus
    corpus = load_corpus()

    # Initialize pipeline
    pipeline = EFSGPipeline(corpus=corpus)

    input_file = input_dir / 'report-requests.jsonl'
    output_file = output_dir / 'run.jsonl'

    if not input_file.exists():
        print(f' ERROR: {input_file} not found')
        sys.exit(1)

    print(f'[TIRA] Reading from: {input_file}')
    print(f'[TIRA] Writing to:   {output_file}')

    topic_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                topic_data = json.loads(line)
                topic = TopicJSON(**topic_data)

                print(f'\n[Topic {line_num}] {topic.topic_id}: {topic.title}')

                report_json, _, _ = pipeline.run(topic, run_id='EFSG_ACL2026')

                f_out.write(json.dumps(report_json, ensure_ascii=False) + '\n')
                f_out.flush()

                topic_count += 1
                print(f'  ✓ Written {len(report_json["responses"])} sentences')

            except Exception as e:
                print(f'  ✗ ERROR: {e}')
                import traceback
                traceback.print_exc()

    print(f'\n[TIRA] ✓ Submission complete: {topic_count} topics processed')
    print(f'[TIRA] Output: {output_file}')


if __name__ == '__main__':
    main()