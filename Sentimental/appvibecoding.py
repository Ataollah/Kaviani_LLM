import os
import json
from typing import Dict, Any, Tuple

import gradio as gr
from openai import OpenAI

# Optional: load .env if available (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


SYSTEM_PROMPT = """You are an expert content safety classifier.

Task:
Given ONE user comment, classify whether it is NSFW or not.

Definition of NSFW for this task:
- Sexual explicit content, pornographic description, fetish content, sexual services solicitation.
- Graphic sexual language intended for arousal.
- Any sexual content involving minors is always NSFW and severe.
- You may also flag extreme graphic gore as NSFW-like, but sexual NSFW has priority.

Output rules:
- Return ONLY valid JSON (no markdown, no extra text).
- Use this exact schema:
{
  "is_nsfw": boolean,
  "confidence": number,          // 0.0 to 1.0
  "category": "sexual" | "sexual_minors" | "graphic_gore" | "none" | "other",
  "severity": "low" | "medium" | "high" | "critical",
  "explanation": string
}

Guidelines:
- If uncertain, lower confidence.
- Keep explanation concise (max 1 sentence).
- For normal safe text, use:
  is_nsfw=false, category="none", severity="low".
"""


def get_client() -> OpenAI:
    api_key = ""
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY environment variable. "
            "Set it before running the app."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            # Recommended by OpenRouter
            "HTTP-Referer": "http://localhost",
            "X-Title": "nsfw-comment-analyzer-gradio",
        },
    )


def extract_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    # Try direct JSON parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract nearest JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw}")


def analyze_comment(comment: str) -> Dict[str, Any]:
    client = get_client()

    response = client.chat.completions.create(
        model="google/gemma-4-31b-it",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Comment to classify:\n{comment}"},
        ],
    )

    raw = response.choices[0].message.content
    data = extract_json(raw)

    # Minimal normalization to keep UI stable
    normalized = {
        "is_nsfw": bool(data.get("is_nsfw", False)),
        "confidence": float(data.get("confidence", 0.0)),
        "category": str(data.get("category", "other")),
        "severity": str(data.get("severity", "low")),
        "explanation": str(data.get("explanation", "")),
    }
    return normalized


def predict(comment: str) -> Tuple[str, Dict[str, Any]]:
    comment = (comment or "").strip()
    if not comment:
        return "⚠️ Please enter a comment.", {}

    try:
        result = analyze_comment(comment)
    except Exception as e:
        return f"❌ Error: {e}", {}

    badge = "🚫 NSFW" if result["is_nsfw"] else "✅ Safe (Not NSFW)"
    summary = (
        f"{badge}\n\n"
        f"Confidence: {result['confidence']:.2f}\n"
        f"Category: {result['category']}\n"
        f"Severity: {result['severity']}\n"
        f"Explanation: {result['explanation']}"
    )
    return summary, result


with gr.Blocks(title="NSFW Comment Analyzer") as demo:
    gr.Markdown("## NSFW Comment Analyzer (OpenRouter + Gemma)")
    gr.Markdown(
        "Enter a text comment, then click **Analyze**.\n\n"
        "Model: `google/gemma-4-31b-it:free` via OpenRouter."
    )

    with gr.Row():
        comment_input = gr.Textbox(
            label="Comment",
            placeholder="Type comment text here...",
            lines=5,
        )

    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

    summary_output = gr.Textbox(label="Result Summary", lines=7)
    json_output = gr.JSON(label="Raw JSON Output")

    analyze_btn.click(
        fn=predict,
        inputs=[comment_input],
        outputs=[summary_output, json_output],
    )

    clear_btn.click(
        fn=lambda: ("", "", {}),
        inputs=[],
        outputs=[comment_input, summary_output, json_output],
    )

    gr.Examples(
        examples=[
            ["I love reading books and drinking coffee."],
            ["Send me explicit adult videos now."],
            ["This movie has extreme gore and dismemberment scenes."],
        ],
        inputs=[comment_input],
    )


if __name__ == "__main__":
    demo.launch()