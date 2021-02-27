from tqdm.notebook import tqdm
from itertools import chain
import json
import re
from app_utils import to_start, rid_blank, preproc_num, to_normal
from trns.preproc_En import pre_en, preproc_en
from trns.preproc_kor import preproc_ko2en


def preProc(X, lang, to_start, pre_ko, preproc_en, en_vocs):
    
    #X = to_start(X)
    
    if lang == 'en':
        #X = pre_en(X)
        X = preproc_en(X,en_vocs)
        X = preproc_num(X)
       
    else:
        X = pre_ko.forward(X)
        X = preproc_num(X) 

    return X 


def preProc_save(X, step_num, lang,to_start, pre_ko, preproc_en, en_vocs,f_toSave):
    
    #n_iter = len(X) // step_num +1
    n_iter = len(X) // step_num + min((len(X) % step_num), 1) * 1

    for i in tqdm(range(n_iter)):
        Xsub = X[i*step_num:(i+1)*step_num]
        #Y = list(chain(*[preProc(s, 'en', to_start, pre_ko, preproc_en, en_vocs) for s in Xsub]))
        Y = preProc(Xsub, lang, to_start, pre_ko, preproc_en, en_vocs)
        with open(f_toSave, 'w' if i==0 else 'a') as f:
            f.write('\n'.join(Y)+'\n')
    return Y

#with open('to_preproc/pre_processed_short05.en', 'r') as f:
#    X = f.read().split('\n')[:100]


def sanitize_input(in_file):
    
    p = re.compile('[\`\^]')
    p1 = re.compile('\`')
    p2 = re.compile('\^')
    
    with open(in_file, 'r') as f:
        X = f.read().split('\n')
        X = [s+'.' for s in X]

    if len([s for s in X if p.search(s) is not None])>0:
        print("`^ id detected !!!!")
        print([(i,s) for i,s in enumerate(X) if p.search(s) is not None])

    X = [p2.sub('ˆ',p1.sub("'",s)) for s in X] #if p.search(s) is None]
    
    return X

def main_proc():
    pre_ko = preproc_ko2en()
    en_vocs = pre_en()

    path0 = 'innovators/'
    f_toSave = path0+'new_output/dev8.'
    step_num = 10000

    X = sanitize_input(path0+'Xen')
    Y = preProc_save(X, step_num, 'en', to_start, pre_ko, preproc_en, en_vocs,f_toSave+'en')

    X = sanitize_input(path0+'Xko')
    Y = preProc_save(X, step_num, 'ko', to_start, pre_ko, preproc_en, en_vocs,f_toSave+'ko')

if __name__ == '__main__':
    main_proc()


"""
with open('innovators/Xen', 'r') as f:
    X = f.read().split('\n')
    X = [s+'.' for s in X]
    
if len([s for s in X if p.search(s) is not None])>0:
    print("`^ id detected !!!!")
    print([(i,s) for i,s in enumerate(X) if p.search(s) is not None])
    
X = [p2.sub('ˆ',p1.sub("'",s)) for s in X] #if p.search(s) is None]
"""  

"""
step_num = 10000
n_iter = len(X) // step_num + min((len(X) % step_num), 1) * 1
for i in tqdm(range(n_iter)):
    Xsub = X[i*step_num:(i+1)*step_num]
    #Y = list(chain(*[preProc(s, 'en', to_start, pre_ko, preproc_en, en_vocs) for s in Xsub]))
    Y = preProc(Xsub, 'en', to_start, pre_ko, preproc_en, en_vocs)
    with open('innovators/dev8.en', 'w' if i==0 else 'a') as f:
        f.write('\n'.join(Y)+'\n')


with open('innovators/Xko', 'r') as f:
    X = f.read().split('\n')
    
if len([s for s in X if p.search(s) is not None])>0:
    print("`^ id detected !!!!")
    print([(i,s) for i,s in enumerate(X) if p.search(s) is not None])
    
X = [p2.sub('ˆ',p1.sub("'",s)) for s in X] #if p.search(s) is None]
"""


"""
step_num = 10000
n_iter = len(X) // step_num + min((len(X) % step_num), 1) * 1
for i in tqdm(range(n_iter)):
    Xsub = X[i*step_num:(i+1)*step_num]
    #Y = list(chain(*[preProc(s, 'en', to_start, pre_ko, preproc_en, en_vocs) for s in Xsub]))
    Y = preProc(Xsub, 'ko', to_start, pre_ko, preproc_en, en_vocs)
    with open('innovators/dev8.ko', 'w' if i==0 else 'a') as f:
        f.write('\n'.join(Y)+'\n')
"""


