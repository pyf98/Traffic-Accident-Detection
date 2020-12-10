configs = dict()
configs['net'] = dict()

# whether to resume
configs['resume'] = False
configs['ckpt_path'] = 'exp/xx/m_epochxx.pt'
configs['new_lr'] = 1e-5        # if resume is True and new_lr is not None, then use this new learning rate

# for network
configs['net']['n_frames'] = 10
configs['net']['cnn_type'] = 'resnet101'            # 'resnet50', 'resnet101', 'resnet152'
configs['net']['cnn_emb_dim'] = 256
configs['net']['cnn_dropout'] = 0.5
configs['net']['num_rnn_layers'] = 1
configs['net']['rnn_bidir'] = False                 # bidirectional or unidirectional
configs['net']['rnn_hidden_size'] = 256
configs['net']['rnn_dropout'] = 0.3

# for training and validation
configs['lr'] = 1e-4
configs['weight_decay'] = 1e-6
configs['n_epochs'] = 100

configs['save_dir'] = 'exp'
configs['device'] = 'cuda'
configs['n_workers'] = 4
configs['image_size'] = 224                             # for resnet
configs['data_root'] = '/home/ubuntu/data_flow'         # path to folders of optical flows
configs['train_batch_size'] = 8
configs['train_list'] = 'dataset/train.txt'
configs['apply_val'] = True
configs['val_batch_size'] = 8
configs['val_list'] = 'dataset/val.txt'
configs['val_display_interval'] = 5
