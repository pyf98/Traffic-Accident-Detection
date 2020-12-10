import os
import time
import shutil
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from data import OpticalFlowDataset
from conf.cnn_flow import configs


class Trainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device(configs['device'])

        ckpt = None
        if configs['resume']:
            ckpt = torch.load(configs['ckpt_path'])
            configs['net'] = ckpt['net_configs']
            self.configs = configs

        # for pre-trained models
        train_transform = transforms.Compose([
            transforms.Resize([configs['image_size'], configs['image_size']]),
            transforms.ToTensor(),
        ])
        train_dataset = OpticalFlowDataset(
            root=configs['data_root'],
            video_list_path=configs['train_list'],
            n_frames=configs['net']['n_frames'],
            transform=train_transform,
            is_train=True
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=configs['train_batch_size'],
            shuffle=True,
            num_workers=configs['n_workers'],
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )

        assert configs['net']['cnn_type'] in ['resnet50', 'resnet101', 'resnet152']
        if configs['net']['cnn_type'] == 'resnet50':
            from networks import resnet50 as ResNet
        elif configs['net']['cnn_type'] == 'resnet101':
            from networks import resnet101 as ResNet
        elif configs['net']['cnn_type'] == 'resnet152':
            from networks import resnet152 as ResNet

        self.model = ResNet(pretrained=True, channel=configs['net']['n_frames'] * 2)
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=configs['lr'], weight_decay=configs['weight_decay'])

        if configs['resume']:
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            tqdm.write(f"[info] Loaded checkpoint from {configs['ckpt_path']}")
            if configs['new_lr'] is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = configs['new_lr']
                tqdm.write(f"[info] Set new lr to {configs['new_lr']}")

        if configs['apply_val']:
            val_transform = transforms.Compose([
                transforms.Resize([configs['image_size'], configs['image_size']]),
                transforms.ToTensor(),
            ])
            val_dataset = OpticalFlowDataset(
                root=configs['data_root'],
                video_list_path=configs['val_list'],
                n_frames=configs['net']['n_frames'],
                transform=val_transform,
                is_train=False
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=configs['val_batch_size'],
                shuffle=False,
                num_workers=configs['n_workers'],
                collate_fn=val_dataset.collate_fn,
                pin_memory=True
            )
            self.display_interval = configs['val_display_interval']

        self.save_dir = os.path.join(configs['save_dir'], time.strftime('flow-cnn_%Y%m%d-%H%M%S', time.localtime()))
        os.makedirs(self.save_dir)
        self.log_file = os.path.join(self.save_dir, 'log_train.txt')
        self.copyscripts(os.path.join(self.save_dir, 'backup_scripts'))

        self.writelog(self.configs)
        self.writelog('=' * 80)

    def copyscripts(self, dest_path):
        """
        Save python scripts.
        Ignore directories such as '__pycache__' and '.idea'.
        """
        shutil.copytree('.', dest_path, ignore=shutil.ignore_patterns('_*', '.*', self.configs['save_dir']))

    def writelog(self, results):
        if not isinstance(results, str):
            results = str(results)
        with open(self.log_file, 'a') as fp:
            fp.write(results + '\n')

    def savemodel(self, save_name):
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'net_configs': self.configs['net']
            },
            save_name
        )
        tqdm.write(f'[Info] Trained model has been saved as {save_name}')

    def train(self):
        for epoch in trange(self.configs['n_epochs']):
            tqdm.write('=' * 20 + f'Epoch {epoch + 1} starts' + '=' * 20)
            average_loss, accuracy = self.train_epoch(epoch)
            log_str = f'Epoch [{epoch + 1:02d}/{self.configs["n_epochs"]}] Train Loss = {average_loss:.5f} ' \
                      f'Train ACC = {accuracy*100:.2f}%'
            self.savemodel(os.path.join(self.save_dir, f'm_epoch{epoch + 1:02d}.pt'))

            if self.configs['apply_val']:
                with torch.no_grad():
                    val_loss, val_acc = self.val_epoch(epoch)
                    log_str += f' Val Loss = {val_loss:.5f} Val ACC = {val_acc * 100:.2f}%'

            self.writelog(log_str)
            tqdm.write(log_str)

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.
        n_correct = 0
        n_samples = 0
        for step, (images, labels) in enumerate(tqdm(self.train_loader)):
            images = images.to(dtype=torch.float32, device=self.device)       # (N, C=20, H, W)
            labels = labels.to(dtype=torch.float32, device=self.device)       # (N,)

            # forward & backward
            out = self.model(images)        # (N, 1), logits before sigmoid

            tqdm.write(str(images.shape))
            tqdm.write(str(labels.shape))
            tqdm.write(str(torch.sigmoid(out[:4, 0])))
            tqdm.write(str(labels[:4]))

            loss = self.criterion(out, labels.unsqueeze(-1))

            tqdm.write(str(loss.item()) + '\n')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            n_correct += ((out.detach().squeeze(1) >= 0.) == labels).sum().item()
            n_samples += labels.shape[0]

        return total_loss / n_samples, n_correct / n_samples

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss = 0.
        n_correct = 0
        n_samples = 0
        for step, (images, labels) in enumerate(tqdm(self.val_loader)):
            images = images.to(dtype=torch.float32, device=self.device)         # (N, C=20, H, W)
            labels = labels.to(dtype=torch.float32, device=self.device)         # (N,)

            # forward
            out = self.model(images)                            # (N, 1), logits before sigmoid
            loss = self.criterion(out, labels.unsqueeze(-1))

            total_loss += loss.item() * labels.shape[0]
            n_correct += ((out.detach().squeeze(1) >= 0.) == labels).sum().item()
            n_samples += labels.shape[0]

            # display
            if (step + 1) % self.display_interval == 0:
                try:
                    print('-' * 40)
                    print(f'[info] Probs: {torch.sigmoid(out).squeeze(-1).cpu().numpy()}')
                    print(f'[info] Label: {labels.cpu().numpy()}')
                except:
                    print(f'[warning] Failed to display validation result.')

        return total_loss / n_samples, n_correct / n_samples


if __name__ == '__main__':
    trainer = Trainer(configs)
    trainer.train()
