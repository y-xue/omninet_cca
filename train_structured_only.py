

import argparse
import os
import torch
import time
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import random
import sys
from tqdm import tqdm
from libs.utils.train_util import *
import re
import pickle
import time

from libs.utils.vqa.vqa import VQA

parser = argparse.ArgumentParser(description='OmniNet training script.')
parser.add_argument('n_iters', type=int, help='Number of iterations to train.')
parser.add_argument('batch_size', type=int, help='List of batch size for each task seperated by comma')
parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
parser.add_argument('--n_workers', default=0, type=int, help='Number of workers to load data')
parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
parser.add_argument('--eval_interval', type=int, help='Interval after which to evaluate on the test/val set.', default=1000)
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')
parser.add_argument('--weight_seed', default=1029, type=int, help='seed for model weight initialization.')
parser.add_argument('--data_seed', default=68, type=int, help='seed for dataloaders.')
parser.add_argument('--data_path', default='/files/yxue/research/allstate/data', help='data path')
parser.add_argument('--structured_folder', default='synthetic_structured_clustering_std2_valAcc66', help='path to structured data.')
parser.add_argument('--model_save_path', default='/out/test', help='path to save the model.')
parser.add_argument('--dropout', default=0.1, type=float, help='default dropout rate')
parser.add_argument('--cca_n_heads', default=8, type=int, help='number of cca heads.')
parser.add_argument('--hidden_dim', default=512, type=int)

args = parser.parse_args()

data_path = args.data_path
coco_images = os.path.join(data_path, 'coco/train_val')
vqa_dir = os.path.join(data_path, 'vqa')
structured_path = os.path.join(data_path, 'vqa', args.structured_folder, 'data.pkl')

with open(os.path.join(data_path, 'vqa', args.structured_folder, 'num_cat_dict.pkl'), 'rb') as f:
    num_cat_dict = pickle.load(f)

input_dim = 6005
output_dim = 3500
hidden_dim = args.hidden_dim

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(int(args.weight_seed))

if not os.path.exists(args.model_save_path):
    os.makedirs(args.model_save_path)

def one_hot_encoding(a, c):
    return np.hstack([one_hot(a[:,i], c[i]) for i in range(a.shape[1])])

def one_hot(a, c):
    b = np.zeros((a.size, c))
    b[np.arange(a.size), a]=1
    return b

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class StructuredDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'struct': torch.tensor(self.X[idx]).float(), 'ans': torch.tensor(self.Y[idx])}
        return sample

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class vqa_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None, structured_data=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.ans = []
        self.structured = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                self.ans.append(ans_to_id[m_a])
                if structured_data is not None:
                    self.structured.append(structured_data[x['question_id']])
        self.transform = transforms
        self.N=len(self.ans)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        ans = self.ans[idx]
        struct = None
        if len(self.structured) != 0:
            struct = torch.as_tensor(self.structured[idx]).float()
        return {'struct': struct, 'ans': ans}


def vqa_batchgen(vqa_dir, image_dir, num_workers=1, batch_size=1, data_seed=68, structured_path=None):
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)

        structured_data = None
        if structured_path is not None:
            with open(structured_path, 'rb') as f:
                structured_data = pickle.load(f)

        # a transformation for the images
        vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
        vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
        vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        # the dataset
        dataset = vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=None, structured_data=structured_data)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_collate_fn, drop_last=True,pin_memory=True)
        print('training mini-batches:', len(dataloader))
        val_dataset = vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=None, structured_data=structured_data)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)
        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader


def vqa_collate_fn(data):
    # the collate function for dataloader
    collate_ans=[]
    collate_struct=[]
    for d in data:
        collate_ans.append((d['ans']))
        collate_struct.append((d['struct']))
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    if len(collate_struct) != 0:
        collate_struct = torch.stack(collate_struct, dim=0)
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'ans': collate_ans,
        'struct': collate_struct
    }


def train(model, dataloader, val_dl):
    criterion = nn.NLLLoss()
    optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.init_lr, betas=(0.9,0.98), eps=1e-09)
    # optimizer = SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.init_lr, momentum=0.9)

    best_acc = 0

    it = -1
    log_str = ''
    
    start_time = time.time()
    for i in range(args.n_iters):
        it += 1
        # optimizer.zero_grad()
        model.zero_grad()
        if i % args.eval_interval == 0 and i > 0:
            log_str += 'training takes {:.2f}s\n'.format(time.time()-start_time)
            start_time = time.time()

            model = model.eval()

            correct = 0
            total = 0
            for b in val_dl:
                x = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float().to(device)
                y = b['ans'].to(device)
                y = y.reshape(-1)

                outputs = model(x)
                predicted = torch.reshape(outputs.argmax(dim=1), [-1]).float()
                loss = criterion(outputs, y)
                total += y.shape[0]
                correct += (predicted == y.float()).sum().cpu().numpy()
            accuracy = 100 * correct / total

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_model.pth'))
                # print('best_iter: %s'%(it))
                log_str += 'best_iter: %s\n'%(it)
            
            log_str += 'Step %d, STRUCT validation loss: %f, Accuracy %f %%\n'%(it, float(loss.detach().cpu().numpy()), accuracy)
            # print('Step %d, STRUCT validation loss: %f, Accuracy %f %%'%(it, float(loss.detach().cpu().numpy()), accuracy))
            log_str += 'validation takes {:.2f}s\n'.format(time.time()-start_time)
            start_time = time.time()
            with open(args.model_save_path + '.log', 'a') as f:
                print(log_str, file=f)
                log_str = ''
            log_str += 'writing log takes {:.2f}s\n'.format(time.time()-start_time)
            start_time = time.time()
            
            model = model.train()

        # start_time = time.time()
        batch = next(dataloader)
        # log_str += 'fetching data takes {:.2f}s\n'.format(time.time()-start_time)
        
        # start_time = time.time()
        x = torch.as_tensor(one_hot_encoding(batch['struct'].int().numpy(), num_cat_dict)).float().to(device)
        y = batch['ans'].to(device)
        y = y.reshape(-1) 
        # print('x', x.shape, x)
        # print('y', y.shape, y)
        # log_str += 'send to device takes {:.2f}s\n'.format(time.time()-start_time)

        # start_time = time.time()
        outputs = model(x)
        predicted = torch.reshape(outputs.argmax(dim=1), [-1]).float()
        
        # print('predicted', predicted.shape, predicted)
        # print('y.reshape(-1)', y.reshape(-1))
        # print('y.argmax(dim=1)', y.argmax(dim=1))
        # raise Exception('debug ends')

        loss = criterion(outputs, y)
        # print('loss', loss) 
        # raise Exception('debug ends')

        n_correct = (predicted == y.float()).sum().cpu().numpy()
        n_total = y.shape[0]
        acc = 100 * (n_correct / n_total)
        # log_str += 'get loss and acc takes {:.2f}s\n'.format(time.time()-start_time)

        # start_time = time.time()
        loss.backward()
        # log_str += 'backward takes {:.2f}s\n'.format(time.time()-start_time)
        # start_time = time.time()
        optimizer.step()
        # log_str += 'optimizer step takes {:.2f}s\n'.format(time.time()-start_time)
        
        log_str += 'Step %d, STRUCT Loss: %f, Accuracy: %f %%\n' % (i, float(loss.detach().cpu().numpy()), acc)
        # print('Step %d, STRUCT Loss: %f, Accuracy: %f %%'%(i,loss.detach(), acc))

    torch.save(model.state_dict(), os.path.join(args.model_save_path, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(args.model_save_path, 'optimizer.pth'))

if __name__ == "__main__":
    # with open('/scratch1/yxuea/data/vqa/synthetic_structured_clustering_std3/synthetic_structured_data_normed.dict', 'rb') as f:
    #     X = pickle.load(f)

    # with open('/scratch1/yxuea/data/vqa/synthetic_structured_clustering_std3/synthetic_structured_data_labels.dict', 'rb') as f:
    #     Y = pickle.load(f)
    # train_Y, val_Y = np.hstack(list(Y['train'].values())), np.hstack(list(Y['val'].values()))
    # train_X, val_X = np.vstack(list(X['train'].values())), np.vstack(list(X['val'].values()))
    # dataset = StructuredDataset(train_X, train_Y)
    # dataloader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    # val_dataset = StructuredDataset(val_X, val_Y)
    # val_dl = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    # print(len(dataloader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FFN(input_dim, output_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(device)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    dataloader, val_dl = vqa_batchgen(vqa_dir, coco_images, num_workers=args.n_workers, batch_size=int(args.batch_size), data_seed=int(args.data_seed+args.restore), structured_path=structured_path)
    train(model, dataloader, val_dl)

"""
class StrcturedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': torch.tensor(self.X[idx]).float(),
                  'y': torch.tensor(self.Y[idx])}

        return sample
"""

