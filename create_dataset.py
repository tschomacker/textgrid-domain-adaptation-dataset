import urllib.request
import zipfile
from tqdm.auto import tqdm
import spacy
import argparse
from random import shuffle

def file_to_sentence_list(nlp,zip_file_object,dataset_file,shuffle_sentences=True):
    print('start reading the file')
    sentences =[]
    for file_name in tqdm(zip_file_object.namelist()):
        # only consider text-files
        if '.txt' in file_name:
            with zip_file_object.open(file_name) as current_file:
                f = open(dataset_file, "a")
                for line_bytes in current_file.readlines():
                    line_str = line_bytes.decode()
                    # skip small paragraphs such as headlines etc.
                    if len(line_str.split(' ')) > 9:
                        text = line_str
                        #sentences = [i for i in nlp(text).sents]
                        for sentence in nlp(text).sents:
                            sentence_clean = str(sentence).replace('\n','').replace('«','"').replace('»','"')
                            sentences.append(sentence_clean)
    print('Successfully read',len(sentences),'sentences from', len(zip_file_object.namelist()), 'files')
    if shuffle_sentences:
        shuffle(sentences)
    with open(dataset_file, "w") as f:
        for sentence in sentences:
            f.write(sentence+'\n')
        f.close()


def main():
    parser = argparse.ArgumentParser(description="Download a TextGrip Zip-File, separate, shuffles it sentences and writes them to a file; one sentence per line")
    parser.add_argument(
        '--spacy_model',
        type=str,
        default="de_core_news_sm",
        help='The name or path of the spacy model.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output/domain_adaptation_sentences.txt',
        help='Name or directory of the output file.'
    )
    parser.add_argument(
        '--input_url',
        type=str,
        default='https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=format%3Atext%2Fxml&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames=',
        help='Textgrid-URL, where the zip-file can be downloaded.'
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help='Whether or not to shuffle the sentences before writing them.'
    )
    
    args = parser.parse_args()
    
    spacy_model_name = args.spacy_model
    
    
    # setup spacy
    spacy.cli.download(spacy_model_name)
    nlp = spacy.load(spacy_model_name)
    
    # Some other options for the url to consider:
    ## url = 'https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=edition.agent.value%3AKafka%2C+Franz&filter=format%3Atext%2Fxml&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames='
    ## url = 'https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=format%3Atext%2Fxml&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames='
    
    url = args.input_url
    print('Retrieve the zip-file from\n', url)
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')

    
    dataset_file = args.output_file
    
    file_to_sentence_list(nlp,zip_file_object,dataset_file, args.shuffle)
            
    
if __name__ == "__main__":
    main()