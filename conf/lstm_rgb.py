configs = dict()
configs['net'] = dict()

# whether to resume
configs['resume'] = True
configs['ckpt_path'] = 'exp/rgb_20201207-101404/m_epoch20.pt'
configs['new_lr'] = 2e-5        # if resume is True and new_lr is not None, then use this new learning rate

# for network
configs['net']['cnn_type'] = 'resnet101'            # 'resnet50', 'resnet101', 'resnet152'
configs['net']['cnn_emb_dim'] = 512
configs['net']['cnn_dropout'] = 0.5
configs['net']['cnn_finetune'] = False              # whether to fine-tune the pre-trained CNN
configs['net']['num_rnn_layers'] = 1
configs['net']['rnn_hidden_size'] = 256
configs['net']['rnn_dropout'] = 0.3
configs['net']['rnn_bidir'] = True                 # bidirectional or unidirectional

# for training and validation
configs['lr'] = 2e-4
configs['weight_decay'] = 1e-6
configs['n_epochs'] = 100

configs['save_dir'] = 'exp'
configs['device'] = 'cuda'
configs['n_workers'] = 4
configs['train_batch_size'] = 64
configs['train_n_frames'] = 15                  # max number of frames per video during training
configs['image_size'] = 224                     # for resnet
configs['data_root'] = '/home/ubuntu/data'      # path to folders of rgb frames
configs['train_list'] = 'dataset/train.txt'
configs['apply_val'] = True
configs['val_batch_size'] = 32
configs['val_n_frames'] = 30
configs['val_list'] = 'dataset/val.txt'
configs['val_display_interval'] = 5
