#!/usr/bin/env python3
"""CLI tool for flower classification predictions."""
import click
import numpy as np
import joblib
from pathlib import Path
import json
import sys


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Flower Classification CLI - Predict flower species from morphological features."""
    pass


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    model_path = Path('models/best_model.joblib')
    preprocessor_path = Path('models/preprocessor.joblib')
    metadata_path = Path('models/metadata.json')
    
    if not model_path.exists():
        click.echo(click.style('Error: Model not found. Please train the model first.', fg='red'))
        sys.exit(1)
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            class_names = metadata.get('class_names', class_names)
    
    return model, preprocessor, class_names


@cli.command()
@click.option('--sepal-length', '-sl', type=float, required=True, help='Sepal length in cm')
@click.option('--sepal-width', '-sw', type=float, required=True, help='Sepal width in cm')
@click.option('--petal-length', '-pl', type=float, required=True, help='Petal length in cm')
@click.option('--petal-width', '-pw', type=float, required=True, help='Petal width in cm')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed prediction info')
def predict(sepal_length, sepal_width, petal_length, petal_width, verbose):
    """Predict flower species from input features."""
    model, preprocessor, class_names = load_model_and_preprocessor()
    
    input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = preprocessor.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    species = class_names[prediction]
    
    click.echo('')
    click.echo(click.style('=' * 50, fg='cyan'))
    click.echo(click.style('       FLOWER CLASSIFICATION RESULT', fg='cyan', bold=True))
    click.echo(click.style('=' * 50, fg='cyan'))
    click.echo('')
    
    click.echo(f'  Sepal Length: {sepal_length} cm')
    click.echo(f'  Sepal Width:  {sepal_width} cm')
    click.echo(f'  Petal Length: {petal_length} cm')
    click.echo(f'  Petal Width:  {petal_width} cm')
    click.echo('')
    click.echo(click.style('Predicted Species: ', bold=True) + click.style(species.upper(), fg='green', bold=True))
    
    if verbose and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        confidence = np.max(proba) * 100
        
        click.echo('')
        click.echo(click.style('Class Probabilities:', bold=True))
        for name, p in zip(class_names, proba):
            bar_len = int(p * 20)
            bar = '#' * bar_len + '-' * (20 - bar_len)
            click.echo(f'  {name:12s}: {p*100:5.1f}% [{bar}]')
        click.echo('')
        click.echo(f'Confidence: {confidence:.1f}%')
    
    click.echo('')


@cli.command()
def info():
    """Show model information."""
    metadata_path = Path('models/metadata.json')
    
    click.echo('')
    click.echo(click.style('=' * 50, fg='cyan'))
    click.echo(click.style('          MODEL INFORMATION', fg='cyan', bold=True))
    click.echo(click.style('=' * 50, fg='cyan'))
    click.echo('')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for key, value in metadata.items():
            if isinstance(value, list):
                click.echo(f'{key}: {", ".join(map(str, value))}')
            else:
                click.echo(f'{key}: {value}')
    else:
        click.echo('Model: Iris Classification Model')
        click.echo('Classes: setosa, versicolor, virginica')
        click.echo('Features: sepal_length, sepal_width, petal_length, petal_width')
    
    click.echo('')


@cli.command()
def demo():
    """Run demo prediction with sample data."""
    click.echo('')
    click.echo(click.style('Running demo prediction...', fg='yellow'))
    click.echo('')
    
    ctx = predict
    ctx.invoke(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
        verbose=True
    )


if __name__ == '__main__':
    cli()
