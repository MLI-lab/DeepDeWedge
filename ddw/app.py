import typer
from .prepare_data import prepare_data
from .fit_model import fit_model
from .refine_tomogram import refine_tomogram


app = typer.Typer()
app.command()(prepare_data)
app.command()(fit_model)
app.command()(refine_tomogram)

def main():
    app()

if __name__ == "__main__":
    main()
