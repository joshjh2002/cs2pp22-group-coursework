'''
ipynb_wordCount.py

Purpose:
    Count the words in a Jupyter notebook file.
    Prints the markdown word count to the screen.

Notes:
    Hard-coded to explore markdown text only.
    No thoughtful argument checking applied.

Source: 
    https://jackmckew.dev/counting-words-with-python.html

Requires:
    pip install nbformat

Run:
    python ipynb_wordCount.py ./path/to/file.ipynb

'''

import io
from typing import List
from nbformat import current
import sys


def count_words_in_jupyter(filePath: str, returnType:str = 'markdown'):
    with io.open(filePath, 'r', encoding='utf-8') as f:
        nb = current.read(f, 'json')

    word_count_markdown: int = 0
    word_count_heading: int = 0
    word_count_code: int = 0
    for cell in nb.worksheets[0].cells:
        if cell.cell_type == "markdown":
            word_count_markdown += len(cell['source'].replace('#', '').lstrip().split(' '))
        elif cell.cell_type == "heading":
            word_count_heading += len(cell['source'].replace('#', '').lstrip().split(' '))
        elif cell.cell_type == "code":
            word_count_code += len(cell['input'].replace('#', '').lstrip().split(' '))

    if returnType == 'markdown':
        print(f'{word_count_markdown=}')
        print(f'{word_count_heading=}')
        print(f'{word_count_code=}\n')
        return word_count_markdown
    elif returnType == 'heading':
        return word_count_heading
    elif returnType == 'code':
        return word_count_code
    else:
        return Exception


if __name__ == "__main__":

    file = sys.argv[1]

    print(f'Words counting toward coursework limit: {count_words_in_jupyter(filePath=file, returnType="markdown")}')