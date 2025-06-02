from ollama import chat
from ollama import ChatResponse
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

def main():
    # ollama.generate(model='llama3.2', prompt='Why is the sky blue?')
    pass

def load_stories(path_name):
    df = pd.read_csv()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('/Users/sperdijk/Documents/PhD/Datasets/Pretraining/chisor_dataset_all/ChiSCor_CoNLL_paper/csv/ChiSCor_master_df_password/ChiSCor_master_df.csv'))

    args = parser.parse_args()

    response: ChatResponse = chat(model='llama3.2:1b', messages=[
    {
        'role': 'user',
        'content': 'Generate a story',
    },
    ])

    print(response.message.content)



