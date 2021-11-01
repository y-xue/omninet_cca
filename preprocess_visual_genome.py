import json
import pickle
import glob
import numpy as np

data_dir = '/files/yxue/research/allstate/data/visual_genome'

def remove_preps(qa):
    for i in range(len(qa)):
        for j in range(len(qa[i]['qas'])):
            ans = qa[i]['qas'][j]['answer']
            words = ans.split(' ')
            if len(words) > 1:
                if len(words) > 2 and ' '.join(words[:2]) in ['On the', 'In the', 'At the', 'To the', 
                                               'On a',   'In a',   'At a',   'To a',
                                               'During the', 'It is']:
                        modified_ans = ' '.join(words[2:])
                elif words[0] in ['A', 'The', 'On', 'In', 'At', 'Light', 'To', 'It\'s']:
                    modified_ans = ' '.join(words[1:])
            if modified_ans[-1] == '.':
                modified_ans = modified_ans[:-1]
            qa[i]['qas'][j]['answer'] = modified_ans
    with open(data_dir+'/question_answers_extracted.json','w') as f:
        json.dump(qa, f)
    

def build_vocab(qa):
    remove_preps(qa)
    
    ans_count = {}
    for x in qa:
        for y in x['qas']:
            ans = y['answer']
            if ans not in ans_count:
                ans_count[ans] = 0
            ans_count[ans] += 1

    ans_to_id = {}
    id_to_ans = {}
    idx = 0
    for ans in dict(sorted(ans_count.items(), key=lambda item: item[1],reverse=True)):
        if len(ans.split(' ')) == 1:
            ans_to_id[ans] = idx
            id_to_ans[idx] = ans
            idx += 1

    with open('conf/vg_vocab.pkl', 'wb') as f:
    	pickle.dump([ans_to_id, id_to_ans], f)

def split_into_val_test():
    np.random.seed(210)
    fns = glob.glob(data_dir+'/VG_100K_2/*')
    ids = [int(fn.split('/')[-1].split('.')[0]) for fn in fns]
    np.random.shuffle(ids)
    val_ids = ids[:(len(ids)//2)]
    test_ids = ids[(len(ids)//2):]
    with open(data_dir+'/val_image_ids.lst.pkl','wb') as f:
        pickle.dump(val_ids, f)
    with open(data_dir+'/test_image_ids.lst.pkl','wb') as f:
        pickle.dump(test_ids, f)

    return val_ids, test_ids

def split_qa(qa, val_ids, test_ids):
    train_qa, val_qa, test_qa = [], [], []
    for i in range(len(qa)):
        if len(qa[i]['qas']) == 0:
            continue
        image_id = qa[i]['qas'][0]['image_id']
        if image_id in val_ids:
            val_qa.append(qa[i]['qas'])
        elif image_id in test_ids:
            test_qa.append(qa[i]['qas'])
        else:
            train_qa.append(qa[i]['qas'])

    with open(data_dir+'/train_qa.json','w') as f:
        json.dump(train_qa, f)
    with open(data_dir+'/val_qa.json','w') as f:
        json.dump(val_qa, f)
    with open(data_dir+'/test_qa.json','w') as f:
        json.dump(test_qa, f)


with open(data_dir+'/question_answers.json','r') as f:
    qa = json.load(f)

build_vocab(qa)
val_ids, test_ids = split_into_val_test()
split_qa(qa, val_ids, test_ids)



