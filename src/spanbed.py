#!/usr/bin/python

from transformers import RobertaTokenizerFast, RobertaModel
from transformers import AutoModel, AutoTokenizer

import torch
import itertools
import sys
import argparse
import csv
from typing import Iterable, Callable, List


"""
A CLI wrapper around transformers to compute contextualized span embeddings for lines in input, yielding .csv output.

A 'contextualized span embedding' is the embedding of a given span of text, but crucially as processed by a model 
that also saw some surrounding text. Concretely, it computes token embeddings for the full text, then averages 
the embeddings of tokens inside the span, ignoring token embeddings outside the span.

Input is a csv of triples sentence,start,stop, or (with --bracket) lines like "this is a sentence and [I'd like to embed this span] and ignore this."

Will use SpanBERT by default, though note that it is designed for subsentential (<10 words) spans. A solid alternative might be roberta-base, or the original Bert.
"""


def main():
    args = parse_args()

    writer = csv.writer(sys.stdout)
    reader = bracketed_reader(args.spans) if args.brackets else csv_reader(args.spans)

    model = make_contextualized_sentence_transformer(args.model, args.hidden)

    for spans in batched(reader, 10000):   # large 'batch', to interfere with transformers' batching as little as possible
        embs = model(spans)
        for emb in embs:
            writer.writerow(emb.tolist())


def bracketed_reader(lines):
    for line in lines:
        line = line.strip()
        start = line.index('[')
        end = line.index(']') - 1
        text = line.replace('[', '').replace(']', '')
        yield {'text': text, 'start': start, 'end': end}


def csv_reader(lines):
    csvreader = csv.DictReader(lines, fieldnames=['text', 'start', 'end'])
    for d in csvreader:
        d['start'], d['end'] = int(d['start']), int(d['end'])
        yield d


def make_contextualized_sentence_transformer(model_name: str, hidden_states_to_use: List[int]) -> Callable:
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    # model = RobertaModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def contextualized_sentence_transformer(spans: Iterable[dict]):
        """
        As input, takes an iterable of dictionaries, each with keys 'text', 'start', and 'end'.
        """
        texts = [t['text'] for t in spans]
        starts = [t['start'] for t in spans]
        ends = [t['end'] for t in spans]
        encoded_input = tokenizer(texts, return_tensors='pt', return_offsets_mapping=True, padding=True)

        # mask any tokens outside the span

        span_mask = [[m and start <= s < end for m, (s, e) in zip(mask, offsets)] for (mask, offsets, start, end) in zip(encoded_input['attention_mask'], encoded_input['offset_mapping'], starts, ends)]
        span_mask_tensor = torch.tensor(span_mask, dtype=torch.bool)   # torch.Size([3, 18])

        # dimensions in comments are for an example batch of 3 sentences, longest 18 tokens, with 4 hidden layers requested.

        output = model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], output_hidden_states=True)

        hidden_states = [output['hidden_states'][h] for h in hidden_states_to_use]  # 4  x  torch.Size([3, 18, 768])
        hidden_states_stacked = torch.stack(hidden_states)  # torch.Size([4, 3, 18, 768])

        # set tokens outside the span to nan:
        hidden_states_stacked_masked = hidden_states_stacked.masked_fill(~span_mask_tensor.unsqueeze(0).unsqueeze(-1), torch.nan) # torch.Size([4, 3, 18, 768])

        # for remaining (non-nan) tokens, average first over hidden states then over tokens.
        mean_hidden_state = torch.nanmean(hidden_states_stacked_masked, dim=0)  # torch.Size([3, 18, 768])
        span_embeddings = mean_hidden_state.nanmean(dim=-2)    # torch.Size([3, 768])

        return span_embeddings

    return contextualized_sentence_transformer

def parse_args():
    parser = argparse.ArgumentParser(description='Script to compute embedding of a span with other text as context.')
    parser.add_argument('spans', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='File containing csv-triples of text,start,stop (or with --brackets: lines like "bla bla [bla bla bla] bla bla"), default stdin. Start and stop must be character offsets.')
    parser.add_argument('--brackets', action='store_true', help='To input spans as "blablabla [bla bla] bla blabla" instead of csv.')
    parser.add_argument('--model', type=str, default='SpanBERT/spanbert-large-cased', help='Embedding model to use; default SpanBERT.')
    parser.add_argument('--hidden', type=str, default='-1', help='Which hidden states to use (comma-separated ints)')
    args = parser.parse_args()

    if args.spans == '-':
        args.spans = sys.stdin
    args.hidden = map(int, args.hidden.split(','))

    return args


# Included here, since only available in Python 3.12...
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


if __name__ == '__main__':
    main()
