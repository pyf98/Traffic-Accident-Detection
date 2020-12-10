configs = dict()
configs['net'] = dict()

# whether to resume
configs['resume'] = False
configs['ckpt_path'] = 'exp/flow_20201202-063020/m_epoch43.pt'
configs['new_lr'] = 1e-4        # if resume is True and new_lr is not None, then use this new learning rate

# for network
configs['net']['cnn_type'] = 'resnet101'            # 'resnet50', 'resnet101', 'resnet152'
configs['net']['n_frames'] = 10                     # number of frames

# for training and validation
configs['lr'] = 1e-3
configs['weight_decay'] = 1e-6
configs['n_epochs'] = 100

configs['save_dir'] = 'exp'
configs['device'] = 'cuda'
configs['n_workers'] = 4
configs['image_size'] = 224                         # for resnet
configs['train_batch_size'] = 64
configs['data_root'] = '/home/ubuntu/data_flow'     # path to folders of optical flow images
configs['train_list'] = 'dataset/train.txt'
configs['apply_val'] = True
configs['val_batch_size'] = 32
configs['val_list'] = 'dataset/val.txt'
configs['val_display_interval'] = 5
