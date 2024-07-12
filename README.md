# SpanBed: Contextualized Span Embeddings #

A command-line wrapper around [transformers](https://huggingface.co/docs/transformers/), to compute _contextualized span 
embeddings_ for lines from the input, yielding the embeddings as .csv output.

A 'contextualized span embedding' is an embedding for a given span of text, _as processed by a model 
that also saw some surrounding text_. Concretely, it computes contextualized token embeddings for the full text, then averages 
the embeddings of the tokens inside the span.

Input is a csv of triples sentence,start,stop, or (with --bracket) lines like "this is a sentence and [I'd like to embed this span] and ignore this."

Will use [SpanBERT](https://huggingface.co/SpanBERT/spanbert-large-cased) by default, though note that it is designed for subsentential (<10 words) spans. A solid alternative might be roberta-base, or the original Bert.

## Install ##

Recommended is to first install pipx for ease of installing Python command line tools:

`pip install pipx`

Then: 

`pipx install git+https://github.com/mwestera/spanbed`

This will make the command `spanbed` available in your shell.

## Usage ##

```bash
$ echo "This is an interesting sentence is it not,11,31" | spanbed > embeddings.csv
```

Or from a file:

```bash
$ spanbed < spans.csv > embeddings.csv
```

Or with a different model, and/or with specific layers (uses only last layer by default):

```bash
$ echo "This is an interesting sentence is it not,11,31" | spanbed --model roberta-base --hidden 8,9,10,11
```

Or using a different, bracketed span notation: 

```bash
$ echo "This is an [interesting sentence] is it not" | spanbed --hidden -2,-1 --brackets
```
