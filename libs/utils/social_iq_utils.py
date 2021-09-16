import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import numpy

def qai_to_tensor(in_put,keys,total_i=1):
    data=dict(in_put.data)
    features=[]
    for i in range(len(keys)):
        features.append(numpy.array(data[keys[i]]["features"]))
    input_tensor=numpy.array(features,dtype="float32")[:,0,...]
    in_shape=list(input_tensor.shape)
    q_tensor=input_tensor[:,:,:,0:1,:,:]
    ai_tensor=input_tensor[:,:,:,1:,:,:]

    return q_tensor,ai_tensor[:,:,:,0:1,:,:],ai_tensor[:,:,:,1:1+total_i,:,:]

def get_judge():
    return nn.Sequential(OrderedDict([
        ('fc0',   nn.Linear(340,25)),
        ('sig0', nn.Sigmoid()),
        ('fc1',   nn.Linear(25,1)),
        ('sig1', nn.Sigmoid())
        ]))

def flatten_qail(_input):
    return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0,2)
    

def build_qa_binary(qa_glove,keys):
    return qai_to_tensor(qa_glove,keys,1)


def build_visual(visual,keys):
    vis_features=[]
    for i in range (len(keys)):
        this_vis=numpy.array(visual[keys[i]]["features"])
        this_vis=numpy.concatenate([this_vis,numpy.zeros([25,2208])],axis=0)[:25,:]
        vis_features.append(this_vis)
    return numpy.array(vis_features,dtype="float32").transpose(1,0,2)

def build_acc(acoustic,keys):
    acc_features=[]
    for i in range (len(keys)):
        this_acc=numpy.array(acoustic[keys[i]]["features"])
        numpy.nan_to_num(this_acc)
        this_acc=numpy.concatenate([this_acc,numpy.zeros([25,74])],axis=0)[:25,:]
        acc_features.append(this_acc)
    final=numpy.array(acc_features,dtype="float32").transpose(1,0,2)
    return numpy.array(final,dtype="float32")

 
def build_trs(trs,keys):
    trs_features=[]
    for i in range (len(keys)):
        this_trs=numpy.array(trs[keys[i]]["features"][:,-768:])
        this_trs=numpy.concatenate([this_trs,numpy.zeros([25,768])],axis=0)[:25,:]
        trs_features.append(this_trs)
    return numpy.array(trs_features,dtype="float32").transpose(1,0,2)
 
def process_data(keys):

    qa_glove=social_iq["QA_BERT_lastlayer_binarychoice"]
    visual=social_iq["DENSENET161_1FPS"]
    transcript=social_iq["Transcript_Raw_Chunks_BERT"]
    acoustic=social_iq["Acoustic"]

    qas=build_qa_binary(qa_glove,keys)
    visual=build_visual(visual,keys)
    trs=build_trs(transcript,keys)  
    acc=build_acc(acoustic,keys)    
    
    return qas,visual,trs,acc

def to_pytorch(_input):
    return Variable(torch.tensor(_input)).cuda()

def reshape_to_correct(_input,shape):
    return _input[:,None,None,:].expand(-1,shape[1],shape[2],-1).reshape(-1,_input.shape[1])

def calc_accuracy(correct,incorrect):
    correct_=correct.cpu()
    incorrect_=incorrect.cpu()
    return numpy.array(correct_>incorrect_,dtype="float32").sum()/correct.shape[0]

