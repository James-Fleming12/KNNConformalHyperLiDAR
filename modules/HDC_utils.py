from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
        super(Model, self).__init__()

        self.device = device

        # Record the current number of class hypervectors
        self.num_classes = num_classes      # Used in supervised HD
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        # set the input dimension
        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
        w_dict = torch.load(modeldir + "/SENet_valid_best",
                            map_location=lambda storage, loc: storage)
        self.net.load_state_dict(w_dict['state_dict'], strict=True)
        self.net.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.net.cuda()

        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            # Generate a random projection matrix
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim, 
                                          randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        
        else:  # No encoder, use raw samples
            self.hd_dim = self.input_dim

        # Set classify
        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        # self.classify_weights is the sum of all hypervectors, so its scale
        # accounts the number of samples in this class/cluster
        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        # print(self.classify_weights.shape)  # size num_class x HD dim

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        # print("x.shape", x.shape)  # torch.Size([1, 5, 64, 512])

        with torch.cuda.amp.autocast(enabled=True):
            x = self.net(x, True)
        
        # print("x.shape", x.shape)  # torch.Size([1, 128, 64, 512])
        # x = self.flatten(x)
        x = x.permute(0, 2, 3, 1)  # shape: (1, 64, 512, 128)
        x = x.reshape(-1, 128)     # shape: (1*64*512, 128) = (32768, 128)
        # sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device)
        # print("x.shape", x.shape)  # torch.Size([32768, 128])
        if PERCENTAGE is not None:
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select
            
            if is_wrong is not None:
                # # Pick by the wrong and keep the PERCENTAGE
                wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
                
                if wrong_indices.numel() >= num_samples:
                    # If there are enough wrong samples, randomly select from them
                    selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                    is_wrong[selected_indices] = False # Mark the selected indices as used
                else:
                    # If there are not enough wrong samples, fill the rest with random samples
                    non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                    remaining = num_samples - wrong_indices.numel()
                    fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]
    
                    selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                    is_wrong[selected_indices] = False # Mark the selected indices as used
            else:
                selected_indices = torch.randperm(x.shape[0], device=x.device)[:num_samples]

            selected_indices, _ = selected_indices.sort()  # Optional: sort to preserve order
            # print("selected_indices", selected_indices.shape)  # e.g., torch.Size([1638])
            x = x[selected_indices]  # shape: (~PERCENTAGE * 32768, 128)
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"

            # Pick by loss: 
            # num_samples = int(x.shape[0] * PERCENTAGE)
            # num_wrongdata = 0
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # top_indices = sorted_indices[:num_wrongdata]

            # all_indices = torch.arange(is_wrong.shape[0], device=x.device)
            # temp = torch.ones_like(is_wrong, dtype=torch.bool)
            # temp[top_indices] = False
            # remaining_indices = all_indices[temp]

            # remaining = num_samples - num_wrongdata
            # if remaining_indices.numel() >= remaining:
            #     random_fill_indices = remaining_indices[torch.randperm(remaining_indices.shape[0])[:remaining]]
            # else:
            #     # If not enough remaining, take all of them
            #     random_fill_indices = remaining_indices
            
            # selected_indices = torch.cat([top_indices, random_fill_indices], dim=0)
            # is_wrong[selected_indices] = 0 # Mark the selected indices as used

            # Get top losses and their indices (descending sort)
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # selected_indices = sorted_indices[:num_samples]  # pick top N
            # is_wrong[selected_indices] = 0.0

            # Filter your data
            # x = x[selected_indices]
            # print("x after selection", x.shape)  # e.g., torch.Size([1638, 128])
            # print("x", x[0])  # e.g., torch.Size([1638])

        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            # print("Encode bind value: ", self.value(x)[:, :, mask].shape)  # btz*size x num_features * hd_dim
            # print("Encode position value: ", self.position.weight[:, mask].shape)  # num_features * hd_dim
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])  # bsz*size x num_features x hd_dim
            sample_hv[:, mask] = functional.multiset(tmp_hv)  # bsz*size x hd_dim

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:  # None encoder, just use the raw sample
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        # print("sample_hv.shape", sample_hv.shape)  # (bsz*size, 1000)
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        # Get logits output
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))

        #logits = torch.div(logits, self.temperature)
        #softmax_logits = F.log_softmax(logits, dim=1)

        return logits, F.normalize(enc), indices, is_wrong_left # enc is still hd_dim, but some elements are 0

    def get_predictions(self, enc):
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:  # self.method == 'BasicHD'
            #class_hv = self.classify_weights / self.classify_sample_cnt
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()

def set_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return Model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

class KNNModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, gauss_rp=True, use_adaptor=True):
        super(KNNModel, self).__init__()

        self.device = device
        self.use_adaptor = use_adaptor

        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"], use_adaptor=self.use_adaptor)

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
            
            if self.ARCH["train"]["pipeline"] == "pointpillar":
                from modules.HDC_cl import PointPillarEncoder

                class _PointPillarEncoder4D(PointPillarEncoder):
                    def forward(self, batch, only_feat=False):
                        return super().forward(batch).unsqueeze(-1).unsqueeze(-1)

                self.net = _PointPillarEncoder4D(
                    in_channels=self.ARCH["train"].get("pointpillar_in_channels", 4),
                    bev_shape=tuple(self.ARCH["train"].get("pointpillar_bev_shape", [512, 512])),
                )

        if self.ARCH["train"]["pipeline"] != "pointpillar":
            w_dict = torch.load(modeldir + "/SENet_valid_best", map_location=lambda storage, loc: storage)
            
            state_dict = w_dict['state_dict']
            model_state = self.net.state_dict()
            for k in list(state_dict.keys()):
                if k in model_state and state_dict[k].shape != model_state[k].shape:
                    del state_dict[k]
                    
            self.net.load_state_dict(state_dict, strict=False)
            self.net.eval()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.gpu = True
                self.net.cuda()
        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            torch.manual_seed(42) # setting fixed seed for projection initialization (removes saved model randomness)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            if not gauss_rp:
                # self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim) 
                    self.projection.weight.copy_(gaussian_matrix / np.sqrt(self.input_dim))
            else:
                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim)
                    q, _ = torch.linalg.qr(gaussian_matrix)
                    self.projection.weight.copy_(q * torch.sqrt(torch.tensor(self.hd_dim))) # Scale by the square root of the dimension to preserve variance (Johnson-Lindenstrauss)

            torch.set_rng_state(torch_rng_state) # set back to random
            np.random.set_state(numpy_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim,  randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        self.gauss_rp = gauss_rp

        self.register_buffer('proto_momentum', torch.zeros_like(self.classify.weight.data)) # EMA momentum

        # KNN Confidence Bank
        self.k = 100
        self.bank_size = 3000
        self.bank = {c: torch.empty((0, self.hd_dim), device=self.device) for c in range(self.num_classes)}

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None, chunk_idx=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast('cuda', enabled=True):
            x = self.net(x, only_feat=True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        if chunk_idx is not None:
            start, end = chunk_idx
            x = x[start:end]

        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select

            if wrong_indices.numel() >= num_samples:
                selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                is_wrong[selected_indices] = False
            else:
                non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                remaining = num_samples - wrong_indices.numel()
                fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]

                selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                is_wrong[selected_indices] = False

            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])
            sample_hv[:, mask] = functional.multiset(tmp_hv)

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))

        return logits, F.normalize(enc), indices, is_wrong_left

    def get_predictions(self, enc):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_confidence(self, enc, preds=None):
        """
        Returns the k-NN contrastive confidence (higher = more trustworthy).
        Score is the negative ratio of in-class distance to out-of-class distance.
        """
        if preds is None:
            preds = self.get_predictions(enc).argmax(dim=1)
            
        enc = F.normalize(enc)
        # We want higher confidence to be better. So we compute - (d_in / d_out)
        # Initialize with -6e4 (a low valid float16 confidence)
        confidence = torch.full((enc.shape[0],), -6e4, device=self.device, dtype=enc.dtype)
        
        for c in preds.unique().tolist():
            m = preds == c
            hc = enc[m]
            
            in_bank = self.bank.get(c, torch.empty(0, device=self.device))
            out_bank = torch.cat([self.bank[o] for o in self.bank if o != c and self.bank[o].shape[0] > 0], dim=0) if len(self.bank) > 1 else torch.empty(0, device=self.device)
            
            if in_bank.shape[0] == 0:
                # If no samples in the bank for this class, we can't trust it.
                continue
                
            if out_bank.shape[0] == 0:
                # If no out-of-class samples exist, we have perfect confidence
                confidence[m] = 0.0
                continue
                
            # Compute d_in (distance to nearest neighbors within predicted class)
            k_in = min(self.k, in_bank.shape[0])
            sims_in = hc @ in_bank.T
            topk_in = sims_in.topk(k_in, dim=1).values
            d_in = (1.0 - topk_in).clamp_min(0.0).mean(dim=1)
            
            # Compute d_out (distance to nearest neighbors across all other classes)
            k_out = min(self.k, out_bank.shape[0])
            sims_out = hc @ out_bank.T
            topk_out = sims_out.topk(k_out, dim=1).values
            d_out = (1.0 - topk_out).clamp_min(0.0).mean(dim=1)
            
            # Ratio (lower ratio = more trustworthy) -> negate for confidence
            confidence[m] = - (d_in / d_out.clamp_min(1e-4))
            
        return confidence

    @torch.no_grad()
    def update_bank(self, enc, labels):
        """
        Add newly admitted, highly confident samples to the k-NN bank.
        Uses a FIFO queue to enforce `self.bank_size`.
        """
        enc = F.normalize(enc)
        for c in labels.unique().tolist():
            m = labels == c
            new_samples = enc[m]
            
            if c not in self.bank:
                self.bank[c] = torch.empty((0, self.hd_dim), device=self.device, dtype=enc.dtype)
                
            # If the dtypes don't match (e.g. from float32 initial empty buffer), cast the bank
            if self.bank[c].dtype != enc.dtype:
                self.bank[c] = self.bank[c].to(dtype=enc.dtype)
                
            self.bank[c] = torch.cat([self.bank[c], new_samples], dim=0)
            
            # Truncate to retain only the most recent samples
            if self.bank[c].shape[0] > self.bank_size:
                self.bank[c] = self.bank[c][-self.bank_size:]
                
    @torch.no_grad()
    def online_update(self, x, learning_rate=0.01, threshold=-1.0):
        """
        Takes an input batch x (cenet image), encodes it, computes k-NN confidence,
        and uses the highly confident samples to update the class prototypes 
        (self.classify.weight) via EMA style updates.
        """
        self.eval()
        enc, _, _ = self.encode(x)
        num_total_samples = enc.shape[0]

        original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
        valid_enc_mask = torch.any(original_x != 0, dim=1) # ignore background
        
        if not torch.any(valid_enc_mask):
            return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        
        active_enc = enc[valid_enc_mask]
        enc_norm = F.normalize(active_enc)
        
        if enc_norm.dtype != self.classify.weight.dtype:
            enc_norm = enc_norm.to(self.classify.weight.dtype)

        logits = self.classify(enc_norm)
        predictions = torch.argmax(logits, dim=1)
        
        full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
        full_predictions[valid_enc_mask] = predictions

        # Compute k-NN confidence (negated ratio of d_in / d_out. Higher is better)
        confidences = self.get_confidence(enc_norm, predictions)
        
        # Filter based on confidence threshold
        valid_mask = confidences > threshold
        
        if not torch.any(valid_mask):
            return full_predictions

        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        unique_classes = torch.unique(predictions[valid_indices])

        for class_id in unique_classes:
            c_id = class_id.item()

            class_mask = (predictions == c_id) & valid_mask
            if not torch.any(class_mask):
                continue
                
            class_indices = torch.nonzero(class_mask).squeeze(1)
            sample_encs = enc_norm[class_indices]
            class_confs = confidences[class_indices]

            # Shift confidences to positive weights using exp() since they are negative ratios
            weights = torch.exp(class_confs)
            weights = weights / weights.sum()
            
            weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
            
            # effective_lr scaled by average confidence multiplier
            effective_lr = learning_rate * torch.exp(class_confs.mean()).item()

            current_weight = self.classify.weight[c_id]
            
            # EMA Momentum Update
            self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
            updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
            
            updated_weight_norm = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
            self.classify.weight[c_id] = updated_weight_norm

        return full_predictions
    
def set_knn_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return KNNModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)