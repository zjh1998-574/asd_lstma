import pyprind
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p_fold=10
num_epochs=100
bath_size=32
result=[]
three_result=[]
result=[]
sub_result=[]
three_result=[]
sub_three_result=[]
result.append(num_epochs)
sub_result.append(num_epochs)
print("num_epochs",num_epochs)
y_arr = np.array([get_label(f,labels) for f in new_flist])
new_flist = np.array(new_flist)

kk = 0
kf = StratifiedKFold(n_splits=p_fold, random_state=22, shuffle=True)
for kk, (train_index, test_index) in enumerate(kf.split(new_flist,y_arr)):
    train_data_flist,test_data_flist=new_flist[train_index],new_flist[test_index]

    train_data,train_label,train_Pheno_Info=data_agument2(all_data=all_data, Pheno_Info=Pheno_information,sample_list=train_data_flist, fix_seq_length=90, crops=10)
    test_data,test_label,test_Pheno_Info = data_agument2(all_data=all_data, Pheno_Info=Pheno_information,sample_list=test_data_flist, fix_seq_length=90, crops=10)
    train_loader = get_loader2(data=train_data,labels=train_label,Pheno_Info=train_Pheno_Info,bath_size=bath_size,mode='train')
    test_loader = get_loader2(data=test_data,labels=test_label,Pheno_Info=test_Pheno_Info,bath_size=bath_size,mode='test')
    model=Self_encoder(200,8,1,90,0.1,0.1)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer =optim.Adam(model.parameters(),lr=0.0001)
    res_mlp= Transformer_encoder_train4(model, num_epochs, train_loader,test_loader,optimizer, criterion, device)
    print(res_mlp)
    result.append(res_mlp[0])
    sub_result.append(res_mlp[3])
    three_result.append(res_mlp[0:3])
    sub_three_result.append(res_mlp[3:6])
print("seq_averages:")
result_mean = np.mean(np.array(result[1:]), axis=0)
print(result_mean)
print("seq_three_result:")
print(np.mean(np.array(three_result), axis=0))
print("sub_averages:")
sub_result_mean = np.mean(np.array(sub_result[1:]),axis=0)
print(sub_result_mean)
print("sub_three_result:")
print(np.mean(np.array(sub_three_result), axis=0))
