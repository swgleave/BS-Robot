#Thanks to the user KarlH on the fast.ai forums for some of this code.  

from fastai.text import *
import html
import torchtext
from torchtext import vocab, data

#reload model and dependencies for future use

PATH=Path('data/BS/')

CLAS_PATH=Path('data/bs_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('data/bs_lm/')
LM_PATH.mkdir(exist_ok=True)

em_sz,nh,nl = 400,1150,3

trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
vs=len(itos)
vs,len(trn_lm)

wd=1e-7
bptt=70
bs=52

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
learner.load('lm1bs25epoch')


TEXT = data.Field(lower=True, tokenize="spacy")

#Function to generate text
def sample_model(m, s, l=50):
    s_toks = Tokenizer().proc_text(s)
    s_nums = [stoi[i] for i in s_toks]
    s_var = V(np.array(s_nums))[None]

    m[0].bs=1
    m.eval()
    m.reset()

    res, *_ = m(s_var)
    print('...', end='')

    for i in range(l):
        r = torch.multinomial(res[-1].exp(), 2)
        #r = torch.topk(res[-1].exp(), 2)[1]
        if r.data[0] == 0:
            r = r[1]
        else:
            r = r[0]
        
        word = itos[to_np(r)[0]]
        res, *_ = m(r[0].unsqueeze(0))
        print(word, end=' ')
    print('\n')
    m[0].bs=bs
    
m=learner.model

ss=input("please enter seed text: ")
while ss != "-1":
    #ss="""That's easy. During the '99 NBA Playoffs, Ewing tore an Achilles tendon during the second game of the Eastern finals against Indiana. """
    #ss="""we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct #the other way """
    
    print(ss,"...")
    sample_model(m,ss,l = 100)
    ss=input("please enter seed text (to quit, enter -1): ")
print('fin!')