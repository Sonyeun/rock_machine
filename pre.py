
#목적: 실시간 데이터에 의한 패턴에 인공지능이 학습할 수 있는가
#->실시간 데이터를 받아들여야 됨
#lstm을 이용한다. 즉, context를 기억하게 한다.

#lstm layer은 input_dim은 3개(가위, 바위, 보)
#inp_shape = (batch_size = 1, seq_len = 1, input_dim = 3)
#out_shape = (batch_size = 1, seq_len = 1, input_dim = 3)
lstm_layer = nn.LSTM(input_size = 3, hidden_size = 3, num_layers = 1, batch_first=False)
optimizer = optim.Adam(lstm_layer.parameters(), lr = 0.1)

#나의 의도는 실시간으로 lstm의 가중치가 조정되도록

for counts in range(fights):
    #이전 데이터를 기반으로 marcof의 수
    #기존 인덱스대로 번호를 구하고, 인덱스 순환시키기
    marcof = mod1.markov_matrix(ai_df, array_ai) 
    marcof_card = mod1.twotonum(marcof)
    marcof_pre = torch.tensor([[[0,0,0]]]).float()
    marcof_pre[0][0][(marcof_card+1)%3] = 1
    marcof = marcof_pre.clone().detach()

    #이전 데이터를 기반으로 ai의 수
    #ai는 output그대로 가져가고, marcof는 
    out, hidden = lstm_layer(marcof_df)
    ai = hidden[0].squeeze()
    ai_card = torch.argmax(ai, dim = 0) #0차원 tensor
    #ai[(int(ai_card)-1)%3] = 0
    #ai[(int(ai_card)+1)%3] = 0
    marcof[0][0][ai_card] = 1

    ######## marcof쪽을 변경해야겠다
    for i in range(3):
        if marcof[0][0][i] == 1:
            marcof[0][0][i] = 0
        else:
            marcof[0][0][i] = 1

    ########
    #대결     
    loss = F.mse_loss(marcof, ai) 
    print('---for loss------')
    print('marcof', marcof)
    print('ai', ai)
    print('ai: %s, marcof: %s' %(mod1.cardtonum(ai_card), mod1.cardtonum(marcof_card)))
    print('loss', loss)

    #marcof 갱신
    if ai_df is not None:
        array_ai[ai_df][ai_card] += 1
    ai_df = ai_card

    #ai 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    marcof_df = marcof_pre

    #print('marcof', marcof)
    #print('marcof_pre', marcof_pre)
    #print('marcof_df', marcof_df)
    
    result = mod1.a_winner(ai_card, marcof_card, mode = False)
    if result == None:
        same += 1
    elif result == True:
        ai_win += 1
    elif result == False:
        ai_lose += 1
    
    #print('ai: %s, marcof: %s' %(mod1.cardtonum(ai_card), mod1.cardtonum(marcof_card)))
    #print('loss', loss)
    if (counts+1)%verbose ==0:
        print('시행횟수: ',counts)
        print('ai 승: %d, marcof 승: %d, 무승부: %d' %(ai_win, ai_lose, same))
        print('ai 승률: %d, marcof 승률: %d' %(ai_win/counts*100, ai_lose/counts*100))
        print('')






#ai는 가위 바위 보
#1이면 0으로, 2면 1로, 0이면 2로
#marcof 보 가위 바위

#대결을 고쳐야 돼!!!
 # 그리고 random도 궁금
 # peace
