configs = dict()
configs['net'] = dict()

# whether to resume
configs['resume'] = False
configs['ckpt_path'] = 'exp/xxxx/m_epochxx.pt'
configs['new_lr'] = 1e-3        # if resume is True and new_lr is not None, then use this new learning rate

# for network
configs['net']['cnn_type'] = 'resnet101'            # 'alexnet'
configs['net']['hidden_sizes'] = [256, 128]
configs['net']['batchnorms'] = [True, True, True]
configs['net']['dropouts'] = [0.3, 0.2, 0.1]   # the first is dropout for input

# for training and validation
configs['lr'] = 1e-2
configs['weight_decay'] = 1e-6
configs['n_epochs'] = 100

configs['save_dir'] = 'exp'
configs['device'] = 'cuda'
configs['n_workers'] = 4
configs['train_batch_size'] = 64 #1024
configs['image_size'] = 224                 # for resnet
configs['data_root'] = '/home/ubuntu/project/data'
configs['train_list'] = 'trainlist_reduced.txt'
configs['apply_val'] = True
configs['val_batch_size'] = 128
configs['val_list'] = 'val.txt'
configs['val_display_interval'] = 5
