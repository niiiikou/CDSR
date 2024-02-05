# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM





class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=768,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        assert module_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nonlinearity in ['relu', 'tanh', 'sigm']

        self.module_type = module_type

        if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(emb_dimension, emb_dimension),
                                          nn.ReLU(),
                                          nn.Linear(emb_dimension, emb_dimension))


        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.module_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.module_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)




class MetaICLModel(object):

    def __init__(self, gpt2="gpt2", logger=None, 
        out_dir=None, fp16=False, local_rank=-1, soft_prefix=False, n_tokens=10, 
        prefix_embed_file=None, task_counts=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1
        # print("$$$$$$$$$$$$$$$$$$$:", device)
        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None
        self.load(gpt2)
        self.soft_prefix = soft_prefix

        self.residual = False
        self.n_tokens = n_tokens
        bottleneck_size = 256
        # Get the vocabulary and print the names of the first few tokens
        # vocabulary = self.tokenizer.get_vocab()
        # print("Token IDs and Token Names:")
        # for token_id in range(10):
        #     token_name = vocabulary[token_id]
        #     print(f"{token_id}: {token_name}")

        # print("model:", self.model)

        # for p in self.model.parameters():
        #     print("p:", p.requires_grad)
        # print("self.model.get_input_embeddings().weight.requires_grad", self.model.get_input_embeddings().weight.shape)
        
        if soft_prefix:
            if task_counts is None:
                self.n_tokens = n_tokens
                # print("self.n_tokens:",self.n_tokens)
            else:
                self.n_tokens = n_tokens * len(task_counts)
                # print("self.n_tokens:",self.n_tokens)
            self.orig_vocab_size = self.model.get_input_embeddings().weight.size(0)
            # print("original vocab size: ", self.model.get_input_embeddings().weight.shape)
            self.model.resize_token_embeddings(self.orig_vocab_size + self.n_tokens)

            
            self.new_vocab_size = self.model.get_input_embeddings().weight.size(0)
            # print("self.new_vocab_size:", self.new_vocab_size)
            assert self.new_vocab_size == self.n_tokens + self.orig_vocab_size
            if prefix_embed_file is not None:
                self.model.set_input_embeddings(torch.load(prefix_embed_file))
            else:
                # print("self.model.get_input_embeddings().weight.data:",self.model.get_input_embeddings().weight)
               
                self.model.get_input_embeddings().weight.data[-self.n_tokens:] = \
                    self.model.get_input_embeddings().weight.data[:self.n_tokens]
            self.model.tie_weights()

                # The line essentially copies the weights of the first self.n_tokens tokens (presumably representing the original vocabulary tokens)
                # to initialize the weights of the newly added soft prefix tokens, which are located at the end of the embedding matrix ([-self.n_tokens:]).
                # This operation is likely performed to provide some initial values for the soft prefix embeddings before fine-tuning.

        #     input_ids = torch.tensor([50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276,
        # 23318])
        #     embedding_layer = self.model.get_input_embeddings()
        #     input_embeddings = embedding_layer(input_ids)

        #     print("input_embeddings", input_embeddings)
        #     # print()
        if self.residual:
                # print("self.model.get_input_embeddings().weight.data[-self.n_tokens:]", self.model.get_input_embeddings().weight.data[-self.n_tokens:].shape)
                # if prefix_MLP :
                
            
            print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
            N = self.model.get_input_embeddings().weight.shape[1]
            # N = self.model.encoder.embed_tokens.weight.shape[1]

                  
            self.prefix_MLP = ResMLP(bottleneck_size=bottleneck_size,
                                                emb_dimension=N,
                                                nonlinearity='relu',
                                                #residual=True
                                                residual=True,
                                                )
            self.prefix_MLP.to(self.device)

                

                
            #      prefix_MLP
            # print("hhh:", prefix_MLP)
            # print("old_embedd:", old_embedd.shape)
            # print("prefix_MLP:", prefix_MLP(old_embedd).shape)

            #     self.prefix_MLP.to(self.device)
            # gggggggggggggggggg
            
                    

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, gpt2="gpt2"):

        model = AutoModelForCausalLM.from_pretrained(gpt2)
        self.model_name = gpt2

        if torch.__version__ == '1.14.0.dev20221208+cu117':
            self.model = torch.compile(model)
        else:
            self.model = model 

    def save(self, step, save_all=False):
        if self.local_rank <= 0:
            if save_all:
                model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                    for key, value in self.model.state_dict().items()}
                torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
                self.logger.info("Saving model parameters at step=%d" % step)
            else:
                torch.save(self.model.get_input_embeddings(), 
                    os.path.join(self.out_dir, "soft_embeddings-{}.pt".format(step)))

    def setup_optimizer(self, optimization, num_training_steps, lr, weight_decay, warmup_steps):
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #         {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        #         {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        # freeze all parameters but soft prefix
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.get_input_embeddings().weight.requires_grad = True

        optimizer_grouped_parameters = [
                {'params': self.model.get_input_embeddings().weight, 'weight_decay': weight_decay}
                # {'params': [p for n, p in self.prefix_MLP.named_parameters()], 'weight_decay': weight_decay}
        ]
        # print("fine tune parameters: ", optimizer_grouped_parameters)

        if optimization=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=lr,
                                  relative_step=False,
                                  warmup_init=False,
                                  weight_decay=weight_decay)
            scheduler = None
        elif optimization.startswith("adamw"):
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=lr,
                              eps=1e-08,
                              weight_decay=weight_decay)
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            if optimization=="adamw":
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_training_steps)
            else:
                raise NotImplementedError()
        elif optimization=="8bit-adam":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters,
                                           lr=lr, betas=(0.9, 0.995))
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            raise NotImplementedError()

        self.optimizer = optimizer
        self.scheduler = scheduler

    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)


    def do_train(self, data, batch_size, num_training_steps, save_period, log_period,
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        dataloader = data.get_dataloader(batch_size, is_training=True)
        n_trainable_params = len([param for param in self.model.parameters() if param.requires_grad])
        n_gpus = torch.cuda.device_count()
        self.logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
            n_trainable_params, len(data), num_training_steps, self.n_gpu))

        global_step = 0
        train_losses = []
        best_accuracy = -1
        stop_training=False

        for epoch in range(num_training_steps):
            # print("epoch: ", epoch)
            
            for batch in dataloader:
                global_step += 1
                # print("batch[0]:", batch[0][0])
                # print("batch[2]:", batch[1][0])
                # print("batch[2]:", batch[2][0])
                #########################
                # print("MLP Parameters:")
                # for name, param in self.prefix_MLP.named_parameters():
                #     print(name, param.data)
                #######################
                input_ids=batch[0].to(self.device)
                attention_mask=batch[1].to(self.device)
                token_type_ids=batch[2].to(self.device)
                if len(batch)==3:
                    labels=None
                else:
                    labels=batch[3].to(self.device)
                # print("old_embedd :", self.model.get_input_embeddings().weight[-self.n_tokens:])
                
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = loss.mean()
                # print("new_embedd :", self.model.get_input_embeddings().weight[-self.n_tokens:])
                
                if torch.isnan(loss).data:
                    print ("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())

                if self.fp16:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.model.get_input_embeddings().weight.grad[:self.orig_vocab_size] = 0

                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()    # We have accumulated enought gradients
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.zero_grad()

                if global_step % log_period == 0:
                    self.logger.info("local rank %d\tglobal step %d\ttrain loss %.2f" % (self.local_rank, global_step, np.mean(train_losses)))
                    train_losses = []

                if global_step % save_period == 0:
                    self.save(global_step)

                if global_step==num_training_steps:
                    break

                # #########################
                # print("Parameters after training:")
                # for name, param in self.prefix_MLP.named_parameters():
                #     print(name, param.data)
                # #######################
            
                    
            if global_step==num_training_steps:
                break

        self.logger.info("Finish training")

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist() 
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, return_nll=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        all_nlls = []
        gt_labels = []
        pred_labels = []
        for idx, dp in enumerate(data.metadata):
            # print("dp:", dp)
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            all_nlls.append(curr_label_losses)
            # print("curr_label_losses:", curr_label_losses)
            gt_labels.append(dp["label"])
            # print("dp_label:", dp["label"])
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            # print("prediction_idx:",prediction_idx)
            pred_labels.append(prediction_idx)
            prediction = dp["options"][prediction_idx]
            # print("prediction", prediction)
            predictions.append(prediction.strip())
            
        
        if return_nll:
            return predictions, all_nlls, np.array(gt_labels), np.array(pred_labels)
        else:
            return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        


        # print("self.model.get_input_embeddings().weight:", self.model.get_input_embeddings().weight[-self.n_tokens:])
        # print("self.model.get_input_embeddings().weight.data", self.model.get_input_embeddings().weight.data)
        


        if self.residual:
            N = self.model.get_input_embeddings().weight.shape[0]
            # Get the original word embedding
            orig_wte = self.model.get_input_embeddings()
            # Create a new embedding layer with the same size
            new_wte = nn.Embedding(orig_wte.weight.size(0), orig_wte.weight.size(1))
            new_wte.to(self.device)
            # Copy the original word embedding to the new layer
            new_wte.weight.data[:N-self.n_tokens].copy_(orig_wte.weight.data[:N-self.n_tokens])
            # Apply the prefix MLP to the last n_tokens of the original word embedding
            new_wte.weight.data[N-self.n_tokens:].copy_(self.prefix_MLP(orig_wte.weight.data[N-self.n_tokens:]))
            # print("okayyyyyyyyyyyyy")
            # Set the new embedding layer as the input embedding of the model
            self.model.set_input_embeddings(new_wte)
        # if self.residual:
        #     N = self.model.get_input_embeddings().weight.shape[0]
        #     r_p_mlp = nn.Parameter(self.model.get_input_embeddings().weight.data[:N-self.n_tokens])
        #     p_mlp = nn.Parameter(self.prefix_MLP(self.model.get_input_embeddings().weight[-self.n_tokens:]))
        #     concatenated_weights = nn.Parameter(torch.cat((r_p_mlp, p_mlp), dim=0))
        #     self.model.set_input_embeddings(concatenated_weights)

     
        # print("outttttt")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()
    
        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



