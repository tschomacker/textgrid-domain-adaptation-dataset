import urllib.request
import zipfile
from tqdm.auto import tqdm
import spacy
import argparse
from random import shuffle, randint
import matplotlib.pyplot as plt
from statistics import median, mean
import numpy as np

def file_to_sentence_list(nlp,zip_file_object,mask_percentage,no_docs,shuffle_sentences=True):
    
    sentence_tuples =[]
    name_list = zip_file_object.namelist()
    shuffle(name_list)
    if name_list is None:
        no_docs = 'all'
    elif(len(name_list)) > no_docs+1:
        name_list=name_list[:no_docs]
    else:
        no_docs = 'all'
       
    print('start reading',no_docs,' docs from the zip-file')
    doc_count = 0
    for file_name in tqdm(name_list):
        # only consider text-files
        if '.txt' in file_name:
            with zip_file_object.open(file_name) as current_file:
                for line_bytes in current_file.readlines():
                    line_str = line_bytes.decode()
                    # skip small paragraphs such as headlines etc.
                    if len(line_str.split(' ')) > 9:
                        text = line_str
                        #sentences = [i for i in nlp(text).sents]
                        for sentence in nlp(text).sents:
                            sentence_clean = str(sentence).replace('\n','').replace('«','"').replace('»','"')
                            sentence_tuples.append( (mask_sentence(sentence_clean, mask_percentage) ,
                                               sentence_clean) )
            doc_count += 1
    print('Successfully read',len(sentence_tuples),'sentences from', len(name_list), 'files')
    if shuffle_sentences:
        shuffle(sentence_tuples)
    return sentence_tuples, doc_count

        
def mask_sentence(sentence:str, mask_percentage):
    masked_words = []
    for word in sentence.split(' '):
        if randint(0, 100)  > mask_percentage:
            masked_words.append(word)
        else:
            # masking
            masked_words.append('<mask>')
    return ' '.join(masked_words)


def write_sentences(data_dir, sentence_tuples, tags, args):
    train_source_path = data_dir+'/train_source.txt'
    train_target_path = data_dir+'/train_target.txt'

    val_source_path = data_dir+'/val_source.txt'
    val_target_path = data_dir+'/val_target.txt'

    test_source_path = data_dir+'/test_source.txt'
    test_target_path = data_dir+'/test_target.txt'

    # clear all files
    for file_name in [train_source_path, train_target_path, 
                      val_source_path, val_target_path, 
                      test_source_path, test_target_path]:
        with open(file_name, "w") as f:
            f.write('')
        f.close()
        
    
    validate_index = int(len(sentence_tuples)/args.validate_percentage)
    test_index = validate_index+int(len(sentence_tuples)/args.test_percentage)

    for masked_sentence, sentence in sentence_tuples[:validate_index]:
        with open(val_source_path, "a") as f:
            f.write(tags[0]+masked_sentence+'\n')
        f.close()
        with open(val_target_path, "a") as f:
            f.write(tags[1]+sentence+'\n')
        f.close()
    for masked_sentence, sentence in sentence_tuples[validate_index:test_index]:
        with open(test_source_path, "a") as f:
            f.write(tags[0]+masked_sentence+'\n')
        f.close()
        with open(test_target_path, "a") as f:
            f.write(tags[1]+sentence+'\n')
        f.close()
    
    for masked_sentence, sentence in sentence_tuples[test_index:]:
        with open(train_source_path, "a") as f:
            f.write(tags[0]+masked_sentence+'\n')
        f.close()
        with open(train_target_path, "a") as f:
            f.write(tags[1]+sentence+'\n')
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
        '--data_dir',
        type=str,
        default='output',
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
    parser.add_argument(
        '--add_tags',
        nargs='+',
        default=['de_DE', 'de_DE'],
        help='First tag is the of the source and the second is of the target data. Leave blank if you dont want to use tags.'
    )
    parser.add_argument(
        '--mask_percentage',
        type=bool,
        default=15,
        help='Percentage of words, that are going to be masked.'
    )
    parser.add_argument(
        '--validate_percentage',
        type=bool,
        default=10,
        help='Percentage of sentence, that will be in the validation dataset.'
    )
    parser.add_argument(
        '--test_percentage',
        type=bool,
        default=10,
        help='Percentage of sentence, that will be in the test dataset.'
    )
    parser.add_argument(
        '--no_docs',
        type=int,
        default=50,
        help='Maximum Number of documents from the url should be randomly sampled for the dataset. Leave blank to sample all.'
    )
    parser.add_argument(
        '--print_statisitics',
        type=bool,
        default=True,
        help='Should the statistics be printed?'
    )
    
    args = parser.parse_args()
    
    spacy_model_name = args.spacy_model
    tags = args.add_tags
    if len(tags)==0:
        tags = ['','']
    elif len(tags)==2:
        tags = [tags[0]+' ',tags[0]+' ']
    else:
        raise ValueError('Number of tags should be 0 or 2')
    
    
    # setup spacy
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        spacy.cli.download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
    url = args.input_url
    
    # Some other options for the url to consider:
    ## url = 'https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=edition.agent.value%3AKafka%2C+Franz&filter=format%3Atext%2Fxml&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames='
    ## url = 'https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=edition.agent.value%3AWilde%2C+Oscar&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames='
    ## url = 'https://textgridlab.org/1.0/aggregator/zip/query?query=*&filter=format%3Atext%2Fxml&filter=work.genre%3Aprose&transform=text&meta=false&only=text/xml&dirnames='
    
    
    print('Retrieve the zip-file from\n', url)
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    sentence_tuples, doc_count = file_to_sentence_list(nlp,zip_file_object,args.mask_percentage, args.no_docs, args.shuffle)
    
    write_sentences(args.data_dir, sentence_tuples, tags, args)
    if args.print_statisitics:
        words_per_sentence = [len(sentence.split()) for _, sentence in sentence_tuples]
        maximum = max(words_per_sentence)
        words_per_sentence_median = median(words_per_sentence)
        percentile_90 = np.percentile(words_per_sentence, 90)
        percentile_99 = np.percentile(words_per_sentence, 99)
        
        plt.hist(words_per_sentence, bins=list(range(1, maximum, 1)),color='grey',histtype='stepfilled')
        
        plt.axvspan(words_per_sentence_median, words_per_sentence_median, color='black')
        plt.axvspan(percentile_90, percentile_90, color='black')
        plt.axvspan(percentile_99, percentile_99, color='black')
        plt.axvspan(maximum, maximum, color='black')
        
        #plt.annotate('median'+str(words_per_sentence_median), (words_per_sentence_median,100))
        plt.text(words_per_sentence_median,100,'median: '+str(int(words_per_sentence_median)),horizontalalignment='right',rotation=90.)
        plt.text(percentile_90,100,'90%: '+str(int(percentile_90)),horizontalalignment='right',rotation=90.)
        plt.text(percentile_99,100,'99%: '+str(int(percentile_99)),horizontalalignment='right',rotation=90.)
        plt.text(maximum,100,'max: '+str(int(maximum)),horizontalalignment='right',rotation=90.)
        plt.xlabel('Number of Words ('+str(sum(words_per_sentence))+' in total)')
        plt.ylabel('Number of Sentences \n('+str(len(sentence_tuples))+' in total, from '+str(doc_count)+' document(s))')
        plt.savefig('output/words_per_sentence.pdf', bbox_inches = 'tight', format='pdf')
    

            
    
if __name__ == "__main__":
    main()