"""Executable stdin/stdout protocol example. It intentionally returns mode=demo.

Replace DemoClassifier with the trained classifier, and load payload['checkpoint_path'].
This script is an integration example, not the real training/inference implementation.
"""

import json
import sys

from agent.demo import DemoClassifier


def main():
    payload = json.load(sys.stdin)
    classifier = DemoClassifier()
    if payload["action"] == "describe":
        output = classifier.describe()
    elif payload["action"] == "predict":
        output = classifier.predict(payload["text"], top_k=payload["top_k"])
    else:
        raise ValueError("unsupported action")
    print(output.model_dump_json())


if __name__ == "__main__":
    main()
