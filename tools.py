def data_agument2(all_data,Pheno_Info,sample_list,fix_seq_length,crops):
    augment_data=[]
    agument_label=[]
    augmented_pheno=[]
    nor_augmented_pheno=[]
    for f in sample_list:
        max_seq_length=all_data[f][0].shape[0]
        range_list=list(range(fix_seq_length+1,int(max_seq_length)))
        random_index=random.sample(range_list,crops)
        for j in range(crops):
            r=random_index[j]
            augment_data.append(all_data[f][0][r-fix_seq_length:r])
            agument_label.append(all_data[f][1])
            augmented_pheno.append(Pheno_Info[f])
    augmented_pheno=np.array(augmented_pheno)
    for i in range(augmented_pheno.shape[0]):
        r_mean = np.sum(np.abs(augmented_pheno[i, :])) / augmented_pheno.shape[1]
        for j in range(augmented_pheno.shape[1]):
            if r_mean == 0:
                augmented_pheno[i, j] = 0
            else:
                augmented_pheno[i, j] = augmented_pheno[i, j] / r_mean
    print("data.shape",np.array(augment_data).shape)
    print("Pheno_Info.shape",np.array(augmented_pheno).shape)
    # print(augmented_pheno)
    return np.array(augment_data),np.array(agument_label),np.array(augmented_pheno)

def get_loader2(data=None,labels=None,Pheno_Info=None,bath_size=8,num_workers=0,mode=None):
    dataset=time_series_Dataset2(data=data,labels=labels,Pheno_Info=Pheno_Info)
    if mode=='train':
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=bath_size,
            shuffle=False,
            num_workers=num_workers
        )
    return data_loader
