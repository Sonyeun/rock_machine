import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Net(nn.Module):
    #신경망 정의 및 초기화
    def __init__(self):
        super().__init__() #super()로 기반 클래스(부모 클래스)를 초기화
        self.nums = 10
        fc1 = nn.Linear(self.nums, 30)
        activation1 = nn.ReLU()
        fc2 = nn.Linear(30, 30)
        activation2 = nn.ReLU()
        fc3 = nn.Linear(3,30)

        self.module = nn.Sequential(
            fc1,
            activation1,
            fc2,
            activation2,
            fc3)


    def forward(self, x):
        self.nums = self.nums
        out = self.module(x)
        return out


def markov_matrix(rival_df, array_a):
    #가위(0), 바위(1), 보(2)
    if rival_df == None:
        return F.one_hot(torch.randint(3, size = (1,)), num_classes = 3).float()
    my_card = torch.tensor((np.argmax(array_a[rival_df])+1)%3) #벡터 차원의 텐서
    my_card = F.one_hot(my_card, num_classes = 3).unsqueeze(dim = 0).float()
    return my_card #3차원

def markov_matrix(rival_df, array_a):
    #가위(0), 바위(1), 보(2)
    if rival_df == None:
        return F.one_hot(torch.randint(3, size = (1,)), num_classes = 3).float()
    my_card = torch.tensor((np.argmax(array_a[rival_df])+1)%3) #벡터 차원의 텐서
    my_card = F.one_hot(my_card, num_classes = 3).unsqueeze(dim = 0).float()
    return my_card #3차원

def twotonum(tensor):
    num = torch.argmax(tensor, dim = 1)
    return int(num[0])

def threetonum(tensor):
    num = torch.argmax(tensor, dim = 2)
    return int(num[0][0])

def a_winner(a,b, mode = True):
    """
    a기준 ~ mode가 true인 경우에는 loss 구하기
          ~ a가 이긴경우만 loss는 0
    """
    #가위(0), 바위(1), 보(2)
    result = None
    if a == 0:
        if b == 0:
            result = None
        elif b == 1:
            result = False
        else:
            result = True
    
    elif a == 1:
        if b == 0:
            result = True
        elif b == 1:
            result = None
        else:
            result = False
    
    elif a == 2:
        if b == 0:
            result = False
        elif b == 1:
            result = True
        else:
            result = None

    if mode == False:
        return result

    if result == True:
        return 0
    else:
        return 1


def numtocard(num):
    if num == 0:
        return '가위'
    elif num == 1:
        return '바위'
    elif num == 2:
        return '보'

def for_rsp_loss(ai, marcof, result):
    """
    ai는 1차원 텐서, marcof는 3차원 텐서
    이기는 경우에는 loss = 0.1
    지는 경우에는 loss = 0.7
    비기는 경우에는 loss = 0.3이게 조절한다
    해당 loss를 바탕으로 가중치 갱신"""
    loss = None
    marcof_num = torch.argmax(marcof, dim = 1)
    ai_num = torch.argmax(ai, dim = 0)
    dummy = torch.tensor([0,0,0]).float()
    torch.tanh(ai[(ai_num-1)%3] + 0.15)
    #ai가 이기는 경우
    if result == True: 
        #dummy[ai_num] = ai[ai_num]  + 0.3
        #dummy[(ai_num+1)%3] = ai[(ai_num+1)%3]  + 0.7 #이거 냈으면 짐
        #dummy[(ai_num-1)%3] = ai[(ai_num-1)%3] + + 0.5#이거 냈으면 비김

        dummy[ai_num] = ai[ai_num] + torch.tanh(ai[ai_num]+ 0.3)
        dummy[(ai_num+1)%3] = ai[(ai_num+1)%3] + torch.tanh(ai[(ai_num+1)%3] + 0.7) #이거 냈으면 짐
        dummy[(ai_num-1)%3] = ai[(ai_num-1)%3] + torch.tanh(ai[(ai_num-1)%3] + 0.5) #이거 냈으면 비김

        #ai[(ai_num+1)%3] -= ai[(ai_num+1)%3]
        #ai[(ai_num-1)%3] -= ai[(ai_num-1)%3]

    #ai가 지는 경우
    elif result == False: 
        #dummy[ai_num] = ai[ai_num] + 0.9
        #dummy[(ai_num+1)%3] = ai[(ai_num+1)%3] + 0.7 #이거 냈으면 비김
        #dummy[(ai_num-1)%3] = ai[(ai_num-1)%3]+ 0.5 #이거 냈으면 이김

        dummy[ai_num] = ai[ai_num] + torch.tanh(ai[ai_num] + 0.9)
        dummy[(ai_num+1)%3] = ai[(ai_num+1)%3] + torch.tanh(ai[(ai_num+1)%3] + 0.7) #이거 냈으면 비김
        dummy[(ai_num-1)%3] = ai[(ai_num-1)%3] + torch.tanh(ai[(ai_num-1)%3]+ 0.5) #이거 냈으면 이김

        #dummy[ai_num] = ai[ai_num] + 1
        #dummy[(ai_num+1)%3] = 0.7 #이거 냈으면 ai가 비김
        #dummy[(ai_num-1)%3] = 0.1 #이거 냈으면 이김
        #ai[(ai_num+1)%3] -= ai[(ai_num+1)%3]
        #ai[(ai_num-1)%3] -= ai[(ai_num-1)%3]

    #ai가 비기는 경우
    elif result == None: 
        #dummy[ai_num] = ai[ai_num] + 0.6
        #dummy[(ai_num+1)%3] = ai[(ai_num+1)%3]+ 0.4 #이거 냈으면 이김
        #dummy[(ai_num-1)%3] =ai[(ai_num-1)%3] + 0.8 #이거 냈으면 짐

        dummy[ai_num] = ai[ai_num] + torch.tanh(ai[ai_num] + 0.6)
        dummy[(ai_num+1)%3] = ai[(ai_num+1)%3] + torch.tanh(ai[(ai_num+1)%3]+ 0.4) #이거 냈으면 이김
        dummy[(ai_num-1)%3] = ai[(ai_num-1)%3] + torch.tanh(ai[(ai_num-1)%3] + 0.8) #이거 냈으면 짐

        #dummy[ai_num] = 0.2
        #dummy[(ai_num+1)%3] = 0.08 #이거 냈으면 ai가 이김
        #dummy[(ai_num-1)%3] = 0.5 #이거 냈으면 짐

        #ai[(ai_num+1)%3] -= ai[(ai_num+1)%3]
        #ai[(ai_num-1)%3] -= ai[(ai_num-1)%3]

    loss = F.mse_loss(ai, dummy)

    return loss


