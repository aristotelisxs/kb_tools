import typer


app = typer.Typer(help="Awesome CLI for LLM-related evaluation tools")

eval_app = typer.Typer(help="Eval commands")
app.add_typer(eval_app, name="evaluation")


@eval_app.command("kb-evaluation")
def run_kb_eval(
    qa_filepath: str,
    save_to_filepath: str,
    question_key: str = "input",
    answer_key: str = "expected_output",
    id_key: str = "ticket_id",
    items_key: str = "items",
):
    from tools.kb_completeness import kb_eval

    kb_eval(qa_filepath, save_to_filepath, question_key, answer_key, id_key, items_key)


if __name__ == "__main__":
    app()
