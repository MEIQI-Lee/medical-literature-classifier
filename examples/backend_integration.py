"""Member 5: framework-neutral calls using the actual local model and ToolResult."""

from agent import get_agent

agent = get_agent()  # Lazy singleton: importing this example does not load the model.


def classify_text_request(text: str, top_k: int = 5) -> dict:
    return agent.classify_text(text, top_k=top_k).to_dict()


def classify_upload_request(content: bytes, filename: str, top_k: int = 5) -> dict:
    return agent.classify_file(content, filename, top_k=top_k).to_dict()


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            classify_text_request("分析高血压患者心血管风险与冠心病的关系"),
            ensure_ascii=False,
            indent=2,
        )
    )
