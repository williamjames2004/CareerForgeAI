"""
run_example.py
Standalone script (no web server needed) that takes a resume file,
runs it through the parser + Claude, and prints/saves the JSON result
shown in the example above.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python run_example.py path/to/resume.pdf
    python run_example.py path/to/resume.docx --num-questions 10 --out result.json
"""

import sys
import json
import argparse

from resume_parser import extract_resume_text, ResumeParseError
from question_generator import generate_interview_questions


def main():
    parser = argparse.ArgumentParser(description="Generate interview questions from a resume.")
    parser.add_argument("resume_path", help="Path to the resume file (.pdf, .docx, or .txt)")
    parser.add_argument("--num-questions", type=int, default=8, help="How many questions to generate (default: 8)")
    parser.add_argument("--out", help="Optional path to save the JSON output to a file")
    args = parser.parse_args()

    try:
        print(f"Reading resume: {args.resume_path}", file=sys.stderr)
        resume_text = extract_resume_text(args.resume_path)

        print("Generating interview questions...", file=sys.stderr)
        result = generate_interview_questions(resume_text, num_questions=args.num_questions)

    except ResumeParseError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": "Unexpected error", "details": str(e)}, indent=2))
        sys.exit(1)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nSaved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
