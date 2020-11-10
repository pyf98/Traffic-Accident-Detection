configs = dict()
configs['net'] = dict()

# whether to resume
configs['resume'] = True
configs['ckpt_path'] = 'exp/20201109-11:50:28-1lstm/m_epoch20.pt'
configs['new_lr'] = None        # if resume is True and new_lr is not None, then use this new learning rate

# for network
configs['net']['cnn_type'] = 'resnet101'            # 'resnet50', 'resnet101', 'resnet152'
configs['net']['cnn_emb_dim'] = 512
configs['net']['cnn_dropout'] = 0.2
configs['net']['num_rnn_layers'] = 1
configs['net']['rnn_hidden_size'] = 512
configs['net']['rnn_dropout'] = 0.2

# for training and validation
configs['lr'] = 1e-4
configs['weight_decay'] = 1e-6
configs['n_epochs'] = 60

configs['save_dir'] = 'exp'
configs['device'] = 'cuda'
configs['n_workers'] = 4
configs['train_batch_size'] = 64
configs['train_max_frames'] = 20            # max number of frames per video during training
configs['image_size'] = 224                 # for resnet
configs['data_root'] = '/home/ubuntu'
configs['train_list'] = 'trainlist_reduced.txt'
configs['apply_val'] = True
configs['val_batch_size'] = 32
configs['val_list'] = 'val.txt'
configs['val_display_interval'] = 5
configs['val_max_frames'] = 40
