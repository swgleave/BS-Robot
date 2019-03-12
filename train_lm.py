#python script to train a word-level language model using transfer learning
#Heavily borrowed from fast.ai's deep learning 2 coursework

#Data is required to be in one or more txt files




#import libraries
from fastai.text import *
import html
import torchtext
from torchtext import vocab, data

# A data folder needs to be specified inside the directory.  There needs to be two folders inside the data folder, train and test.  
PATH=Path('data/')

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

#specify path for learning model data
LM_PATH=Path('data/lm/')
LM_PATH.mkdir(exist_ok=True)

#The test and train folders both need to contain at least one additional file, depending on how many classes are in the dataset.  In my use case, there are not 
#multiple classes, so there is one folder in my file stucture inside the train and test folders, called "zero"

#Function that looks inside the data folder, and puts text and label in numpy array format.  If only one set of labels, all will be labeled as a zero.
CLASSES = ['zero']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')

#Add columns names
col_names = ['labels','text']

#shuffle the data.  If data in one text file, nothing should be changed
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]

#store text data in pandas df
df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

#combine train/val data and do train-test split
trn_texts,val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts,val_texts]), test_size=0.1)

#create df from train/test arrays and write to csv
df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)

#define chunksize for tokenzation process
chunksize=24000

re1 = re.compile(r'  +')

#define helper functions for cleaning and tokenizing data
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

#read data back from csv
df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)

#call cleaning/tokenizing function
tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

(LM_PATH/'tmp').mkdir(exist_ok=True)

#save tokenized arrays
np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)

#load numpy arrays
tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')

#compute frequency of words in training corpus
freq = Counter(p for o in tok_trn for p in o)

#set max_vocab and min_freq
max_vocab = 60000
min_freq = 5

#only keep most frequent, using min_freq as min count and max_vocab as max number of words
itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

#create reverse mapping of words in corpus to ints
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

#map words to ints and put in numpy array
trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

#save arrays and mapping dictionary
np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))

vs=len(itos)

#Here we bring in the pretrained model.  First need to download here- http://files.fast.ai/models/wt103/ 

#specify embedding size, hidden units, and number of layers to match pretrained model
em_sz,nh,nl = 400,1150,3

#specify path for pretrained models (need to put downloaded model in this path)
PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

#load weights
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

#some weights in tranfered model will be unknown.  We will want to compute the mean of weights and will set these weights to the mean
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)

#load precomputed weight matrix
itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

#create matrix of zeros to transfer the weights.  If the word exists, transfer it, otherwise use the mean of the weights
new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m

#overwrite weight matrix
wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

#set parameters for language model
wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

#define loaders, and model data using fast.ai objects
trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

#define dropout
drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

#define learner, freeze all layers except last
learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)

#load in pretrained weights
learner.model.load_state_dict(wgts)

#set learning rate
lr=1e-3
lrs = lr

#tune last layer for one epoch
learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)

#unfreeze layers and find learning rate
learner.unfreeze()
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

#train for five additional epochs
learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=5)

#save model and encoder
learner.save('lm1bs')
learner.save_encoder('lm1_encbs')
