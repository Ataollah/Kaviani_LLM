"""
Tehran History Article Writer
Multi-stage LangChain pipeline:
  Stage 1 → Web search (DuckDuckGo, English sources)
  Stage 2 → Summarise & compose English article
  Stage 3 → Translate article to Persian (Farsi)
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
search_tool = DuckDuckGoSearchRun()

# ── Stage 1: Web Search ────────────────────────────────────────────────────────
SEARCH_QUERIES = [
    "Tehran history ancient origins Ray city",
    "Tehran capital Iran Qajar dynasty history",
    "Tehran 20th century modernization Pahlavi era",
    "Tehran historical landmarks bazaar mosques",
    "Tehran city culture heritage civilization",
]

def run_web_search(_input: dict) -> dict:
    console.print(Panel("🔍 [bold cyan]Stage 1 — Web Search[/bold cyan]"))
    all_results = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        for query in SEARCH_QUERIES:
            t = p.add_task(f"  Searching: {query}", total=None)
            try:
                result = search_tool.run(query)
                all_results.append(f"### Query: {query}\n{result}")
            except Exception as exc:
                all_results.append(f"### Query: {query}\n[Error: {exc}]")
            p.remove_task(t)
    return {"raw_search_results": "\n\n---\n\n".join(all_results)}

# ── Stage 2: Summarise & Write English Article ─────────────────────────────────
SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert historian. Using the web-search snippets, write a "
     "well-structured ~800-word article titled 'A Journey Through Tehran's History' "
     "with sections: introduction, ancient origins, Qajar era, Pahlavi modernisation, "
     "contemporary city, and conclusion. Use only facts from the snippets."),
    ("human", "Snippets:\n\n{raw_search_results}\n\nWrite the article now."),
])

def summarise_stage(data: dict) -> dict:
    console.print(Panel("✍️  [bold yellow]Stage 2 — Write English Article[/bold yellow]"))
    chain = SUMMARISE_PROMPT | llm | StrOutputParser()
    article_en = chain.invoke({"raw_search_results": data["raw_search_results"]})
    return {"article_en": article_en, **data}

# ── Stage 3: Translate to Persian ─────────────────────────────────────────────
TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional Persian (Farsi) translator. Translate the article into "
     "fluent, literary Persian. Preserve headings and paragraph structure. "
     "Output ONLY the Persian translation."),
    ("human", "{article_en}"),
])

def translate_stage(data: dict) -> dict:
    console.print(Panel("🌐 [bold magenta]Stage 3 — Translate to Persian[/bold magenta]"))
    chain = TRANSLATE_PROMPT | llm | StrOutputParser()
    article_fa = chain.invoke({"article_en": data["article_en"]})
    return {"article_en": data["article_en"], "article_fa": article_fa}

# ── Pipeline (LCEL) ────────────────────────────────────────────────────────────
pipeline = (
    RunnableLambda(run_web_search)
    | RunnableLambda(summarise_stage)
    | RunnableLambda(translate_stage)
)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    console.print(Panel.fit("[bold]Tehran History Article Writer[/bold]\n"
                            "LangChain + OpenAI + DuckDuckGo", border_style="bright_blue"))
    result = pipeline.invoke({})

    console.print(Panel(result["article_en"], title="[yellow]English Article[/yellow]", border_style="yellow"))
    console.print(Panel(result["article_fa"], title="[magenta]مقاله فارسی[/magenta]", border_style="magenta"))

    with open("tehran_article_en.txt", "w", encoding="utf-8") as f: f.write(result["article_en"])
    with open("tehran_article_fa.txt", "w", encoding="utf-8") as f: f.write(result["article_fa"])
    console.print("\n[bold green]✅ Saved: tehran_article_en.txt & tehran_article_fa.txt[/bold green]")

if __name__ == "__main__":
    main()