# ukr_tokenizer_cli.py
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from tokenizer_utils import (
    load_local_tokenizer,
    preprocess_text,
    split_text_into_words,
    tokenize_text,
)

app = typer.Typer(
    add_completion=True,
    help="Ukrainian tokenizer CLI: test tokenization, inspect tokens, and compute basic metrics.",
)
console = Console()


def _ensure_tokenizer(path: str):
    tok, name = load_local_tokenizer(path)
    if tok is None:
        raise typer.Exit(code=1)
    return tok, name


def _read_lines(file_path: Path, limit: int) -> List[str]:
    lines: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(lines) >= limit:
                break
            s = preprocess_text(line.rstrip("\n"))
            if s.strip():
                lines.append(s)
    return lines


def _tokens_to_strings(tokenizer, ids: List[int]) -> List[str]:
    # tokenizers.Tokenizer supports id_to_token
    out = []
    for tid in ids:
        try:
            out.append(tokenizer.id_to_token(tid))
        except Exception:
            out.append(str(tid))
    return out


@app.command()
def tokenize(
    text: str = typer.Argument(..., help="Input text to tokenize (wrap in quotes)."),
    tokenizer_path: str = typer.Option(
        "ukr_bpe_tokenizer",
        "--tokenizer-path",
        "-t",
        help="Path to tokenizer directory or tokenizer.json",
    ),
    show_tokens: bool = typer.Option(True, "--show-tokens/--no-show-tokens", help="Print token strings."),
    show_ids: bool = typer.Option(True, "--show-ids/--no-show-ids", help="Print token ids."),
    max_display: int = typer.Option(200, "--max-display", help="Max tokens to display."),
):
    """
    Tokenize a single piece of text and print tokenization details.
    """
    tokenizer, name = _ensure_tokenizer(tokenizer_path)

    text_n = preprocess_text(text)
    ids = tokenize_text(tokenizer, text_n)
    toks = _tokens_to_strings(tokenizer, ids)

    words = split_text_into_words(text_n)
    fertility = (len(ids) / len(words)) if words else 0.0
    tok_per_char = (len(ids) / max(len(text_n), 1)) if text_n else 0.0
    chars_per_tok = (len(text_n) / len(ids)) if ids else 0.0

    print(f"[bold]Tokenizer:[/bold] {name}")
    print(f"[bold]Chars:[/bold] {len(text_n)}   [bold]Words:[/bold] {len(words)}   [bold]Tokens:[/bold] {len(ids)}")
    print(f"[bold]Fertility (tok/word):[/bold] {fertility:.3f}")
    print(f"[bold]Tokens/char:[/bold] {tok_per_char:.4f}   [bold]Chars/token:[/bold] {chars_per_tok:.3f}")

    display_n = min(len(ids), max_display)
    if display_n < len(ids):
        print(f"[yellow]Showing first {display_n} tokens (of {len(ids)} total)[/yellow]")

    if show_ids:
        print("\n[bold]Token IDs:[/bold]")
        print(ids[:display_n])

    if show_tokens:
        print("\n[bold]Tokens:[/bold]")
        print(toks[:display_n])


@app.command()
def decode(
    ids: List[int] = typer.Argument(..., help="Token IDs to decode, e.g. 12 345 678"),
    tokenizer_path: str = typer.Option("ukr_bpe_tokenizer", "--tokenizer-path", "-t"),
):
    """
    Decode token IDs back to text (if supported by the local tokenizer).
    """
    tokenizer, name = _ensure_tokenizer(tokenizer_path)

    try:
        text = tokenizer.decode(ids)
    except Exception as e:
        print(f"[red]Decode failed:[/red] {e}")
        raise typer.Exit(code=1)

    print(f"[bold]Tokenizer:[/bold] {name}")
    print(text)


@app.command()
def repl(
    tokenizer_path: str = typer.Option("ukr_bpe_tokenizer", "--tokenizer-path", "-t"),
):
    """
    Interactive prompt: type text and see tokenization; type /quit to exit.
    """
    tokenizer, name = _ensure_tokenizer(tokenizer_path)
    print(f"[bold]Loaded:[/bold] {name}")
    print("Type text to tokenize. Commands: /quit, /help")

    while True:
        try:
            s = console.input("[bold cyan]>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        s = s.strip()
        if not s:
            continue
        if s in ("/quit", "/exit"):
            print("Bye.")
            return
        if s == "/help":
            print("Commands: /quit, /exit, /help")
            continue

        text_n = preprocess_text(s)
        ids = tokenize_text(tokenizer, text_n)
        toks = _tokens_to_strings(tokenizer, ids)

        words = split_text_into_words(text_n)
        fertility = (len(ids) / len(words)) if words else 0.0

        table = Table(title=f"Tokens ({len(ids)}) | Fertility tok/word={fertility:.3f}")
        table.add_column("#", justify="right")
        table.add_column("id", justify="right")
        table.add_column("token")

        for i, (tid, tok) in enumerate(zip(ids[:200], toks[:200]), start=1):
            table.add_row(str(i), str(tid), tok)

        if len(ids) > 200:
            table.caption = "Showing first 200 tokens"

        console.print(table)


@app.command()
def stats(
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="UTF-8 text file, one sample per line. If omitted, reads from stdin.",
    ),
    limit: int = typer.Option(10000, "--limit", "-n", help="Max lines to read."),
    tokenizer_path: str = typer.Option("ukr_bpe_tokenizer", "--tokenizer-path", "-t"),
):
    """
    Compute basic corpus stats: tokens/word and tokens/char on a local file or stdin.
    """
    tokenizer, name = _ensure_tokenizer(tokenizer_path)

    if input_file is not None:
        if not input_file.exists():
            print(f"[red]File not found:[/red] {input_file}")
            raise typer.Exit(code=1)
        texts = _read_lines(input_file, limit)
    else:
        # Read stdin lines
        texts = []
        for line in sys.stdin:
            if len(texts) >= limit:
                break
            s = preprocess_text(line.rstrip("\n"))
            if s.strip():
                texts.append(s)

    if not texts:
        print("[red]No texts provided.[/red]")
        raise typer.Exit(code=1)

    total_tokens = 0
    total_words = 0
    total_chars = 0
    fert_list = []

    for t in texts:
        ids = tokenize_text(tokenizer, t)
        w = split_text_into_words(t)
        total_tokens += len(ids)
        total_words += len(w)
        total_chars += len(t)
        if len(w) > 0:
            fert_list.append(len(ids) / len(w))

    mean_tok_word = (total_tokens / total_words) if total_words else 0.0
    tok_per_char = (total_tokens / total_chars) if total_chars else 0.0
    chars_per_tok = (total_chars / total_tokens) if total_tokens else 0.0

    table = Table(title=f"Corpus stats ({len(texts)} samples) | {name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total samples", str(len(texts)))
    table.add_row("Total chars", f"{total_chars:,}")
    table.add_row("Total words", f"{total_words:,}")
    table.add_row("Total tokens", f"{total_tokens:,}")
    table.add_row("Mean tokens/word", f"{mean_tok_word:.4f}")
    table.add_row("Tokens/char", f"{tok_per_char:.6f}")
    table.add_row("Chars/token", f"{chars_per_tok:.4f}")

    console.print(table)


@app.command()
def export(
    input_file: Path = typer.Argument(..., help="UTF-8 text file, one sample per line."),
    output_jsonl: Path = typer.Argument(..., help="Output JSONL path."),
    tokenizer_path: str = typer.Option("ukr_bpe_tokenizer", "--tokenizer-path", "-t"),
    limit: int = typer.Option(10000, "--limit", "-n"),
):
    """
    Tokenize each line and write JSONL with {text, ids, tokens, token_count, word_count}.
    """
    tokenizer, name = _ensure_tokenizer(tokenizer_path)
    if not input_file.exists():
        print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(code=1)

    texts = _read_lines(input_file, limit)
    if not texts:
        print("[red]No texts loaded.[/red]")
        raise typer.Exit(code=1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for t in texts:
            ids = tokenize_text(tokenizer, t)
            toks = _tokens_to_strings(tokenizer, ids)
            words = split_text_into_words(t)
            rec = {
                "text": t,
                "ids": ids,
                "tokens": toks,
                "token_count": len(ids),
                "word_count": len(words),
                "fertility_tok_per_word": (len(ids) / len(words)) if words else None,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[bold]Wrote:[/bold] {output_jsonl}  ([bold]{len(texts)}[/bold] lines)")
    print(f"[bold]Tokenizer:[/bold] {name}")


def main():
    app()


if __name__ == "__main__":
    main()
