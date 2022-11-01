import torch, torchaudio

class MixIT_dev_data(torch.utils.data.Dataset):

    def __init__(self, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        self.noise_fnames2 = []
        self.start_times2 = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))
                self.noise_fnames2.append(toks[4])
                self.start_times2.append(float(toks[5]))

    def __len__(self):

        return len(self.clean_fnames)

    def __getitem__(self, idx):
        noise_wav, _ = torchaudio.load(self.noise_path + '/' + self.noise_fnames[idx], frame_offset = int(16000 * self.start_times[idx]), 
                                       num_frames = self.wav_len)
        noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
        noise_wav = torch.squeeze(noise_wav)

        noise_wav2, _ = torchaudio.load(self.noise_path + '/' + self.noise_fnames2[idx], frame_offset = int(16000 * self.start_times2[idx]), 
                                        num_frames = self.wav_len)
        noise_wav2 = (noise_wav2 - torch.mean(noise_wav2)) / torch.std(noise_wav2)
        noise_wav2 = torch.squeeze(noise_wav2)

        clean_wav, sfreq = torchaudio.load(self.clean_path + '/' + self.clean_fnames[idx])
        if sfreq != 16000:
            resampler = torchaudio.transforms.Resample(sfreq, 16000)
            clean_wav = resampler(clean_wav)
        clean_wav = torch.squeeze(clean_wav)
        if clean_wav.size(0) < self.wav_len:
            clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
        elif clean_wav.size(0) > self.wav_len:
            clean_wav = clean_wav[:self.wav_len]
        clean_wav = clean_wav - torch.mean(clean_wav)

        factor = 10. ** (self.SNRs[idx] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav2 ** 2))
        clean_wav = clean_wav * factor

        noisy_wav = clean_wav + noise_wav2
        noisy_wav /= torch.std(noisy_wav)

        mixture_wav = noisy_wav + noise_wav
        mixture_wav /= torch.std(mixture_wav)

        noise_stft = torch.stft(input = noise_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        noisy_stft = torch.stft(input = noisy_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        mixture_stft = torch.stft(input = mixture_wav,
                                  n_fft = self.frame_len,
                                  hop_length = self.frame_len // 4,
                                  window = torch.hamming_window(self.frame_len),
                                  return_complex = True)

        feat = torch.abs(mixture_stft) ** 1/15
        feat = feat[None, :, :].to(torch.float32)

        return feat, mixture_stft[None, :, :], noise_stft[None, :, :], noisy_stft[None, :, :]


class PU_dev_data(torch.utils.data.Dataset):

    def __init__(self, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        self.noise_fnames2 = []
        self.start_times2 = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))
                self.noise_fnames2.append(toks[4])
                self.start_times2.append(float(toks[5]))
                
    def __len__(self):

        return len(self.clean_fnames) * 2

    def __getitem__(self, idx):

        if idx % 2 == 1: # The examples with odd indices are noises (considered to be positive examples)
            noise_wav, _ = torchaudio.load(self.noise_path + '/' + self.noise_fnames[idx//2], frame_offset = int(16000 * self.start_times[idx//2]), 
                                           num_frames = self.wav_len)
            noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
            noise_wav = torch.squeeze(noise_wav)

            # extract features 
            noise_stft = torch.stft(input = noise_wav,
                                    n_fft = self.frame_len,
                                    hop_length = self.frame_len // 4,
                                    window = torch.hamming_window(self.frame_len),
                                    return_complex = True)
            mixture_stft = noise_stft
            feat = torch.abs(noise_stft) ** 1/15
            feat = feat[None, :, :].to(torch.float32)

            # compute mask consisting of all 1's
            mask = torch.ones(feat.size(), dtype = torch.float32)
            mixture_wav = noise_wav
            clean_wav = torch.zeros(mixture_wav.size(), dtype = torch.float32)

        else: # The examples with even indices are noisy speeches (considered to be unlabelled examples)
            noise_wav, _ = torchaudio.load(self.noise_path + '/' + self.noise_fnames2[idx//2], frame_offset = int(16000 * self.start_times2[idx//2]), 
                                            num_frames = self.wav_len)
            noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
            noise_wav = torch.squeeze(noise_wav)

            # load clean speech 
            clean_wav, sfreq = torchaudio.load(self.clean_path + '/' + self.clean_fnames[idx//2])
            if sfreq != 16000:
                resampler = torchaudio.transforms.Resample(sfreq, 16000)
                clean_wav = resampler(clean_wav)
            clean_wav = torch.squeeze(clean_wav)
            if clean_wav.size(0) < self.wav_len:
                clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
            elif clean_wav.size(0) > self.wav_len:
                clean_wav = clean_wav[:self.wav_len]
            clean_wav = clean_wav - torch.mean(clean_wav)

            # scale clean speech
            factor = 10. ** (self.SNRs[idx//2] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav ** 2))
            clean_wav = clean_wav * factor

            # mix clean speech and noise
            mixture_wav = clean_wav + noise_wav
            mixture_wav /= torch.std(mixture_wav)
 
            # extract features
            mixture_stft = torch.stft(input = mixture_wav,
                                      n_fft = self.frame_len,
                                      hop_length = self.frame_len // 4,
                                      window = torch.hamming_window(self.frame_len),
                                      return_complex = True)
            feat = torch.abs(mixture_stft) ** 1/15
            feat = feat[None, :, :].to(torch.float32)

            # compute mask consisting of all 0's (indicating unlabelled data)
            mask = torch.zeros(feat.size(), dtype = torch.float32)

        return feat, mask, mixture_stft[None, :, :], mixture_stft[None, :, :]


class PN_data(torch.utils.data.Dataset):

    def __init__(self, partition, frame_len, wav_len, fname, clean_path, noise_path):

        super().__init__()
        self.partition = partition
        self.frame_len = frame_len
        self.wav_len = wav_len
        self.clean_path = clean_path
        self.noise_path = noise_path

        self.clean_fnames = []
        self.noise_fnames = []
        self.start_times = []
        self.SNRs = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                toks = line.split(' ')
                self.clean_fnames.append(toks[0])
                self.noise_fnames.append(toks[1])
                self.start_times.append(float(toks[2]))
                self.SNRs.append(float(toks[3]))

    def __len__(self):

        return len(self.clean_fnames)

    def __getitem__(self, idx):

        # load noise
        noise_wav, _ = torchaudio.load(self.noise_path + '/' + self.noise_fnames[idx], frame_offset = int(16000 * self.start_times[idx]), 
                                       num_frames = self.wav_len)
        noise_wav = (noise_wav - torch.mean(noise_wav)) / torch.std(noise_wav)
        noise_wav = torch.squeeze(noise_wav)

        clean_wav, sfreq = torchaudio.load(self.clean_path + '/' + self.clean_fnames[idx])
        if sfreq != 16000:
            resampler = torchaudio.transforms.Resample(sfreq, 16000)
            clean_wav = resampler(clean_wav)
        clean_wav = torch.squeeze(clean_wav)
        if clean_wav.size(0) < self.wav_len:
            clean_wav = torch.cat([clean_wav, torch.zeros(self.wav_len - clean_wav.size(0))], dim = 0)
        elif clean_wav.size(0) > self.wav_len:
            clean_wav = clean_wav[:self.wav_len]
        clean_wav = clean_wav - torch.mean(clean_wav)

        # scale clean speech
        factor = 10. ** (self.SNRs[idx] / 20.) / torch.sqrt(torch.sum(clean_wav ** 2) / torch.sum(noise_wav ** 2))
        clean_wav = clean_wav * factor

        # mix clean speech and noise
        mixture_wav = clean_wav + noise_wav
        dnm = torch.std(mixture_wav)
        mixture_wav /= dnm
        clean_wav /= dnm
        noise_wav /= dnm

        # extract feature
        mixture_stft = torch.stft(input = mixture_wav,
                                  n_fft = self.frame_len,
                                  hop_length = self.frame_len // 4,
                                  window = torch.hamming_window(self.frame_len),
                                  return_complex = True)
        mixture_stft = mixture_stft[None, :, :]
        feat = torch.abs(mixture_stft) ** 1/15.
        feat = feat.to(torch.float32)

        # compute mask
        clean_stft = torch.stft(input = clean_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        noise_stft = torch.stft(input = noise_wav,
                                n_fft = self.frame_len,
                                hop_length = self.frame_len // 4,
                                window = torch.hamming_window(self.frame_len),
                                return_complex = True)
        mask = torch.abs(clean_stft) > torch.abs(noise_stft)
        mask = mask[None, :, :].to(torch.float32)

        if self.partition == 'train':
            return feat, mask, mixture_stft, clean_stft[None, :, :]
        else:
            return feat, mixture_stft, mixture_wav, clean_wav


def load_data(max_batch_size, world_size, rank, method, frame_len, wav_len, 
              dev_fname, val_fname, test_fname, clean_path, noise_path):

    if method == 'PN':
        train_data = PN_data('train', frame_len, wav_len, dev_fname, clean_path, noise_path)
    elif method == 'MixIT':
        train_data = MixIT_dev_data(frame_len, wav_len, dev_fname, clean_path, noise_path)
    else:
        train_data = PU_dev_data(frame_len, wav_len, dev_fname, clean_path, noise_path)
    val_data = PN_data('val', frame_len, wav_len, val_fname, clean_path, noise_path)
    test_data = PN_data('test', frame_len, wav_len, test_fname, clean_path, noise_path)

    train_batch_size = min(len(train_data) // world_size, max_batch_size)
    val_batch_size = min(len(val_data) // world_size, max_batch_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = True)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = train_batch_size,
                                               shuffle = False,
                                               num_workers = 2,
                                               pin_memory = True,
                                               sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = 2,
                                             pin_memory = True,
                                             sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size = 1,
                                              shuffle = False,
                                              num_workers = 2,
                                              pin_memory = True)
 
    return train_loader, val_loader, test_loader, train_batch_size
