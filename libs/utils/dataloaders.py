
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""
Authors: Subhojeet Pramanik, Priyanka Agrawal, Aman Hussain, Sayan Dutta

Dataloaders for standard datasets used in the paper

"""
import os
import torch
import pickle
import cv2
import numpy as np
import json
import tqdm
import random
from sklearn.model_selection import train_test_split
from PIL import Image
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# the following is required for the lib to work in terminal env
import matplotlib

matplotlib.use("agg", force=True)
from .cocoapi.coco import COCO


from .vqa.vqa import VQA

try:
    from mmsdk import mmdatasdk
except:
    pass

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, data_dir, output_dir, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = data_dir, output_dir
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.resize_height = 300
        self.resize_width = 300
        self.crop_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # The following three parameters are chosen as described in the paper section 4.1

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists('conf/hmdblabels.txt'):
            with open('conf/hmdblabels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels).unsqueeze(0)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True
    
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    
    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in tqdm.tqdm(os.listdir(self.root_dir)):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            split_file=os.path.join('conf/hmdb','%s_test_split1.txt'%file)
            train_files=[]
            test_files=[]
            with open(split_file,'r') as f:
                lines=f.readlines()
                for l in lines:
                    f_name,split=l.strip().split(' ')
                    if split=='1' or split=='0':
                        train_files.append(f_name)
                    elif split=='2':
                        test_files.append(f_name)
            train_dir = os.path.join(self.output_dir, 'train', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
  
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train_files:
                self.process_video(video, file, train_dir)

            for video in test_files:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')
        
        
    def normalize(self, buffer):
        buffer=buffer/255
        for i, frame in enumerate(buffer):
            frame -= np.array([[[0.485, 0.456, 0.406]]])
            frame /= np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer
    
    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] - clip_len>0 and self.split=='train':
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        else:
            time_index=0
        # Randomly select start indices in order to crop the video
        if self.split=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index=0
            width_index=0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer
    
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()


    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer


def hmdb_batchgen(data_dir,process_dir,num_workers=1,batch_size=1,test_batch_size=1,clip_len=16):
        dataset=VideoDataset(data_dir, process_dir, split='train',dataset='hmdb',clip_len=clip_len)
        test_dataset=VideoDataset(data_dir, process_dir, split='test',dataset='hmdb',clip_len=clip_len)
        dataloader=DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                   )
        itr = iter(cycle(dataloader))
        test_dl= DataLoader(test_dataset, num_workers=num_workers, batch_size=test_batch_size,
                                      drop_last=False)
        return itr,test_dl

class coco_cap_dataset(Dataset):
    def __init__(self, ann_file, image_dir,transforms=None,max_words=40):
        caption = COCO(ann_file)
        self.inputs = []
        self.outputs = []
        ann_ids = caption.getAnnIds()
        for idx, a in tqdm.tqdm(enumerate(ann_ids),'Loading MSCOCO to memory'):
            c = caption.loadAnns(a)[0]
            words = c['caption']
            if len(words.split(' '))<=max_words:
                img_file = os.path.join(image_dir, '%012d.jpg' % (c['image_id']))
                self.inputs.append(img_file)
                self.outputs.append(words)
        self.transform = transforms
        self.N=len(self.outputs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.inputs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        cap = self.outputs[idx]
        # returns the dictionary of images and captions
        return {'img': img, 'cap': cap}


def coco_cap_batchgen(caption_dir, image_dir,num_workers=1, batch_size=1):
        # transformations for the images
        train_ann=os.path.join(caption_dir,'captions_train2017.json')
        val_ann=os.path.join(caption_dir,'captions_val2017.json')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = coco_cap_dataset(train_ann, image_dir, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn=coco_collate_fn, drop_last=True,pin_memory=True)
       
        # the iterator over data loader
        itr = iter(cycle(dataloader))
        
        val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
        val_dataset = coco_cap_dataset( val_ann, image_dir, transforms=val_tfms,max_words=5000)
        val_dl= DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2),
                                     collate_fn=coco_collate_fn, drop_last=False,pin_memory=True)
        return itr, val_dl
                    

    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def coco_collate_fn(data):
    # collate function for the data loader
    collate_images = []
    collate_cap = []
    max_len = 0
    for d in data:
        collate_images.append(d['img'])
        collate_cap.append(d['cap'])
    collate_images = torch.stack(collate_images, dim=0)
    # return a dictionary of images and captions
    return {
        'img': collate_images,
        'cap': collate_cap
    }



class vqa_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None, structured_data=None, non_ma_only=False, two_ques_file=None, aug_text_file=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        if two_ques_file is not None:
            with open(two_ques_file, 'rb') as f:
                two_ques_dict = pickle.load(f)

        self.aug_text_file = aug_text_file
        if aug_text_file is not None:
            with open(aug_text_file, 'rb') as f:
                aug_text_dict = pickle.load(f)
        self.imgs = []
        self.ques = []
        self.ques_ids = []
        self.ans = []
        self.ans_str = []
        self.structured = []
        n_features = len(structured_data[list(structured_data.keys())[0]]) if structured_data is not None else 0
        print('n_features:', n_features)
        # self.is_mas = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if (m_a in ans_to_id and not non_ma_only) or (m_a not in ans_to_id and non_ma_only):
                # if m_a in ans_to_id:
                #     self.is_mas.append(True)
                # else:
                #     self.is_mas.append(False)
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                if two_ques_file is not None:
                    # while True:
                    #     another_ques = ques[np.random.randint(len(ques))]
                    #     if another_ques['question_id'] != x['question_id'] and another_ques['image_id'] != x['image_id']:
                    #         break
                    # another_words = another_ques['question']
                    # if np.random.random() < 0.5:
                    #     words = words + ' ' + another_words
                    # else:
                    #     words = another_words + ' ' + words
                    words = two_ques_dict[x['question_id']]
                if aug_text_file is not None:
                    words = [words] + aug_text_dict[x['question_id']]['aug_ques']
                self.ques.append(words)
                if m_a in ans_to_id:
                    self.ans.append(ans_to_id[m_a])
                else:
                    self.ans.append(0)
                self.ans_str.append(m_a)
                self.ques_ids.append(x['question_id'])
                if structured_data is not None:
                    if x['question_id'] in structured_data:
                        self.structured.append(structured_data[x['question_id']])
                    else:
                        self.structured.append(np.random.randint(0,1000,n_features))
        print('ques[7252]', self.ques[7252])
        print('ques[-1]', self.ques[-1])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        ques = self.ques[idx]
        if self.aug_text_file is not None:
            ques = np.random.choice(ques)
        ans = self.ans[idx]
        ans_str = self.ans_str[idx]
        ques_id = self.ques_ids[idx]
        # is_ma = self.is_mas[idx]
        struct = None
        if len(self.structured) != 0:
            struct = torch.as_tensor(self.structured[idx]).float()
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'ques': ques, 'ans': ans, 'ques_id': ques_id, 'struct': struct, 'ans_str': ans_str} #, 'is_ma': is_ma}


def vqa_batchgen(vqa_dir, image_dir, num_workers=1, batch_size=1, structured_path=None, data_seed=68, two_ques_file=None, aug_text_file=None):
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
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer, structured_data=structured_data, two_ques_file=two_ques_file, aug_text_file=aug_text_file)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms, structured_data=structured_data, two_ques_file=two_ques_file)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)

        val_non_ma_dataset = vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms, structured_data=structured_data, non_ma_only=True, two_ques_file=two_ques_file)
        # the data loader
        val_non_ma_dataloader = DataLoader(val_non_ma_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)

        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader,val_non_ma_dataloader


def vqa_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ques_ids = []
    collate_ans=[]
    collate_ans_str=[]
    # collate_is_mas=[]
    collate_struct = []
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ques_ids.append(d['ques_id'])
        collate_ans.append((d['ans']))
        collate_ans_str.append(d['ans_str'])
        # collate_is_mas.append(d['is_ma'])
        if d['struct'] is not None:
            collate_struct.append((d['struct']))
    if len(collate_struct) != 0:
        collate_struct = torch.stack(collate_struct, dim=0)
        
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans,
        'ans_str': collate_ans_str,
        'ques_id': collate_ques_ids,
        'struct': collate_struct
    }

class penn_dataset(Dataset):
    ''' Pytorch Penn Treebank Dataset '''

    def __init__(self, text_file,max_len=150):
        self.X = list()
        self.Y = list()
        with open(text_file) as f:
            # first line ignored as header
            f = f.readlines()[1:]
            for i in range(0,len(f),2):
                if len(f[i].split(' ', maxsplit=1)[1].split(' '))<max_len:
                    self.X.append(f[i])
                    self.Y.append(f[i+1])      
            assert len(self.X) == len(self.Y),\
            "mismatch in number of sentences & associated POS tags"
            self.count = len(self.X)
            del(f)
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return (self.X[idx].split(' ', maxsplit=1)[1],
                self.Y[idx].split()[1:])

    
class PennCollate:
    def __init__(self,vocab_file):
        with open(vocab_file,'r') as f:
            data=json.loads(f.read())
        self.tag_to_id=data['tag_to_id']
        self.id_to_tag=data['id_to_tag']
        
        
    def __call__(self,batch):
        pad_token=self.tag_to_id['<PAD>']
        text=[]
        tokens=[]
        max_len=0
        for b in batch:
            text.append(b[0].strip())
            tok=[self.tag_to_id[tag] for tag in b[1]]
            max_len=max(max_len,len(tok))
            tokens.append(tok)
        for i in range(len(tokens)):
            for j in range(max_len-len(tokens[i])):
                tokens[i].append(pad_token)
        tokens=torch.tensor(np.array(tokens))
        pad_mask=tokens.eq(pad_token)
        #Add padding to the tokens
        return {'text':text,'tokens':tokens,'pad_mask':pad_mask,'pad_id':pad_token}
    
def penn_dataloader(data_dir, batch_size=1, test_batch_size=1,num_workers=8,vocab_file='conf/penn_vocab.json'):
        train_file=os.path.join(data_dir,'train.txt')
        val_file=os.path.join(data_dir,'dev.txt')
        test_file=os.path.join(data_dir,'test.txt')
        collate_class=PennCollate(vocab_file)
        dataset=penn_dataset(train_file)
        val_dataset=penn_dataset(val_file)
        test_dataset=penn_dataset(test_file)
        train_dl=DataLoader(dataset,num_workers=num_workers,batch_size=batch_size,collate_fn=collate_class)
        val_dl=DataLoader(val_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        test_dl=DataLoader(test_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        train_dl=iter(cycle(train_dl))
        return train_dl,val_dl,test_dl
    

class social_iq_dataset(Dataset):
    def __init__(self, data_dir, video_folder, split_dict, split='train', clip_len=30, h=300, w=300):
        self.clip_len = clip_len
        self.split = split
        self.resize_height = h
        self.resize_width = w
        self.crop_size = 224
        
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames = []
        self.ques = []
        self.ans = []
        self.labels = []

        with open(os.path.join(data_dir,'qa.dict.pkl'), 'rb') as f:
            qa = pickle.load(f)

        for d in qa.values():
            if d['video_name'] == 'deKPBy_uLkg_trimmed-out' or (
                split != 'test' and split_dict[d['video_name']] != split):
                # deKPBy_uLkg_trimmed-out is too short
                continue

            vname = d['video_name']
            if split == 'test':
                vname = vname[:-4] + '_trimmed'

            self.fnames.append(os.path.join(data_dir, video_folder, vname))
            self.ques.append(d['question'])
            self.ans.append(d['answer'])
            self.labels.append(d['label'])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return {'video': torch.from_numpy(buffer), 'ques': self.ques[index], 
                'ans': self.ans[index], 'label': self.labels[index]}
        # return torch.from_numpy(buffer), torch.from_numpy(labels).unsqueeze(0)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        buffer=buffer/255
        for i, frame in enumerate(buffer):
            frame -= np.array([[[0.485, 0.456, 0.406]]])
            frame /= np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer
    
    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] - clip_len>0 and self.split=='train':
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        else:
            time_index=0
        # Randomly select start indices in order to crop the video
        if self.split=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index=0
            width_index=0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    # def __init__(self, keys, data, structured_data=None):
    #     self.keys = keys
    #     self.data = data
    #     self.structured_data = structured_data
    #     self.N=len(self.data)

    # def __len__(self):
    #     return self.N

    # def __getitem__(self, idx):
    #     keys = self.keys[idx]
    #     q,a,i=[d[keys[0]:keys[1]] for d in self.data[0]]
    #     vis=self.data[1][:,keys[0]:keys[1],:]
    #     trs=self.data[2][:,keys[0]:keys[1],:]
    #     acc=self.data[3][:,keys[0]:keys[1],:]

def social_iq_batchgen(data_dir, video_folder, num_workers=1, batch_size=1, structured_path=None, clip_len=30, data_seed=68):
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    structured_data = None
    if structured_path is not None:
        with open(structured_path, 'rb') as f:
            structured_data = pickle.load(f)

    with open(os.path.join(data_dir, 'train/split.dict.pkl'), 'rb') as f:
        split_dict = pickle.load(f)

    dataset = social_iq_dataset(data_dir+'/train', video_folder, split_dict, split='train', clip_len=clip_len)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                 collate_fn=social_iq_collate_fn, drop_last=True,pin_memory=True)
    
    print('# of training mini-batches:', len(dataloader))
    val_dataset = social_iq_dataset(data_dir+'/train', video_folder, split_dict, split='val', clip_len=clip_len)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=max(int(batch_size/2),1), shuffle=True,
                                 collate_fn=social_iq_collate_fn, drop_last=False)
    
    test_dataset = social_iq_dataset(data_dir+'/test', video_folder, split_dict, split='test', clip_len=clip_len)
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=max(int(batch_size/2),1), shuffle=True,
                                 collate_fn=social_iq_collate_fn, drop_last=False)
    
    itr = iter(cycle(dataloader))
    return itr,val_dataloader, test_dataloader
    # return itr, test_dataloader


def social_iq_collate_fn(data):
    collate_videos = []
    collate_ques = []
    collate_ans=[]
    collate_labels=[]
    for d in data:
        collate_videos.append(d['video'])
        collate_ques.append(d['ques'])
        collate_ans.append((d['ans']))
        collate_labels.append((d['label']))
        
    collate_videos = torch.stack(collate_videos, dim=0)
    collate_labels=torch.tensor(collate_labels).reshape([-1,1])
    return {
        'videos': collate_videos,
        'ques': collate_ques,
        'ans': collate_ans,
        'labels': collate_labels
    }

# def social_iq_batchgen(data_dir, num_workers=1, batch_size=1, structured_path=None, data_seed=68):
#     structured_data = None
#         if structured_path is not None:
#             with open(structured_path, 'rb') as f:
#                 structured_data = pickle.load(f)

#     # data_dir='/files/yxue/research/allstate/data/socialiq_and_deployed'
#     paths["QA_BERT_lastlayer_binarychoice"]=data_dir+"/socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
#     paths["DENSENET161_1FPS"]=data_dir+"/deployed/SOCIAL_IQ_DENSENET161_1FPS.csd"
#     paths["Transcript_Raw_Chunks_BERT"]=data_dir+"/deployed/SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd"
#     paths["Acoustic"]=data_dir+"/deployed/SOCIAL_IQ_COVAREP.csd"
#     social_iq=mmdatasdk.mmdataset(paths)
#     social_iq.unify()

#     preload=True
#     trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
#     #This video has some issues in training set
#     bads=['f5NJQiY9AuY','aHBLOkfJSYI']
#     folds=[trk,dek]
#     for bad in bads:
#         for fold in folds:
#             try:
#                 fold.remove(bad)
#             except:
#                 pass

#     preloaded_train=process_data(trk)
#     preloaded_dev=process_data(dek)
#     print("Preloading Complete")

#     dataset = social_iq_dataset(trk, preloaded_train, structured_data=structured_data)
#     dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
#                                  collate_fn=social_iq_collate_fn, drop_last=True,pin_memory=True)
    
#     # val_dataset = vqa_dataset(trk[:100], preloaded_dev, structured_data=structured_data)
#     # val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
#     #                              collate_fn=social_iq_collate_fn, drop_last=False)
    

#     test_dataset = vqa_dataset(dek, preloaded_dev, structured_data=structured_data)
#     test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
#                                  collate_fn=social_iq_collate_fn, drop_last=False)
    
#     itr = iter(cycle(dataloader))
#     # return itr,val_dataloader, test_dataset
#     return itr, test_dataset



class vg_dataset(Dataset):
    def __init__(self, qa_file, image_dir, vocab_file, transforms=None):
        with open(qa_file, 'r') as f:
            qas = json.load(f)
        self.imgs = []
        self.ques = []
        self.ques_ids = []
        self.ans = []
        self.ans_str = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f)
        # load the questions
        for x in tqdm.tqdm(qas,'Loading VG data to memory'):
            # get the path
            for qa in x:
                answer = qa['answer']
                img_file = os.path.join(image_dir, '%s.jpg'%qa['image_id'])
                self.imgs.append(img_file)
                self.ques.append(qa['question'])
                self.ans.append(ans_to_id[answer])
                self.ans_str.append(answer)
                self.ques_ids.append(qa['qa_id'])
            
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        ques = self.ques[idx]
        
        ans = self.ans[idx]
        ans_str = self.ans_str[idx]
        ques_id = self.ques_ids[idx]
        
        return {'img': img, 'ques': ques, 'ans': ans, 'ques_id': ques_id, 'ans_str': ans_str}


def vqa_batchgen(vg_dir, image_dir, num_workers=1, batch_size=1, data_seed=68):
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)

        # a transformation for the images
        vg_train_qa=os.path.join(vg_dir,'train_qa.json')
        vg_val_qa=os.path.join(vg_dir,'val_qa.json')
        vg_test_qa=os.path.join(vg_dir,'test_qa.json')
        vocab_file=os.path.join('conf/vg_vocab.pkl')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = vqa_dataset(vg_train_qa, image_dir+'/VG_100K', vocab_file, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = vqa_dataset(vg_val_qa, image_dir+'/VG_100K_2', vocab_file, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)

        test_dataset = vqa_dataset(vg_test_qa, image_dir+'/VG_100K_2', vocab_file, transforms=val_tfms)
        # the data loader
        test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_collate_fn, drop_last=False)

        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader,test_dataloader


def vg_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ques_ids = []
    collate_ans=[]
    collate_ans_str=[]
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ques_ids.append(d['ques_id'])
        collate_ans.append((d['ans']))
        collate_ans_str.append(d['ans_str'])
        
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans,
        'ans_str': collate_ans_str,
        'ques_id': collate_ques_ids
    }





