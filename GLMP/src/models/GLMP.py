import torch
import torch.nn as nn
import random
import numpy as np
import os

from src.utils.evaluation import calc_f1, calc_bleu, calc_distinct
from src.models.loss import masked_cross_entropy
from src.models.modules import ContextRNN, ExternalKnowledge, LocalMemoryDecoder


class GLMP(nn.Module):
    def __init__(self, index2word, vocab_size=30000, hidden_size=512, embed_dim=512, max_resp_len=50, n_layers=1, hop=1,
                 dropout=0.2, teacher_forcing_ratio=0.5,
                 use_cuda=True, use_record=True, unk_mask=True):
        super(GLMP, self).__init__()
        self.index2word = index2word
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.max_resp_len = max_resp_len
        self.n_layers = n_layers
        self.hop = hop
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_cuda = use_cuda
        self.use_record = use_record
        self.unk_mask = unk_mask

        self.encoder = ContextRNN(self.vocab_size, self.hidden_size, self.dropout,
                                  n_layers=self.n_layers, use_cuda=self.use_cuda)
        self.extKnow = ExternalKnowledge(self.vocab_size, self.embed_dim,
                                         self.hop, self.dropout)
        self.decoder = LocalMemoryDecoder(self.vocab_size, self.encoder.embedding, self.index2word, self.hidden_size,
                                          self.dropout, use_cuda=self.use_cuda, use_record=self.use_record)

        self.criterion_bce = nn.BCELoss()

        if self.use_cuda:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

    def load(self, save_dir, file_prefix):
        model_file = "{}/{}.model".format(save_dir, file_prefix)
        if os.path.isfile(model_file):
            state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(model_file))
        else:
            print("Invalid model state file: '{}'".format(model_file))

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing=False, get_decoded_words=False):
        # Build unknown mask for memory
        if self.unk_mask and (not get_decoded_words):
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
            rand_mask = torch.Tensor(rand_mask)
            conv_rand_mask = torch.Tensor(conv_rand_mask)
            if self.use_cuda:
                rand_mask = rand_mask.cuda()
                conv_rand_mask = conv_rand_mask.cuda()
            story = data['context_arr'] * rand_mask.long()
            conv_story = data['conv_arr'] * conv_rand_mask.long()
        else:
            story = data['context_arr']
            conv_story = data['conv_arr']

        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])

        global_pointer, kb_readout = self.extKnow.load_memory(story,
                                                              data['kb_arr_lengths'],
                                                              data['conv_arr_lengths'],
                                                              dh_hidden,
                                                              dh_outputs)
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            copy_list.append(elm_temp)

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            self.extKnow,
            story.size(),
            data['context_arr_lengths'],
            copy_list,
            encoded_hidden,
            data['sketch_response'],
            max_target_length,
            batch_size,
            global_pointer,
            use_teacher_forcing=use_teacher_forcing,
            get_decoded_words=get_decoded_words,
        )

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer

    def iterate(self, data, optimizer=None, grad_clip=None, is_training=True):
        # Encode and Decode
        #use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        max_target_length = max(data['response_lengths'])
        decoder_outputs_vocab, decoder_outputs_ptr, _, _, global_pointer = self.encode_and_decode(
            data, max_target_length,
            use_teacher_forcing=True,
            get_decoded_words=False
        )

        # Loss calculation
        # the binary cross-entropy
        loss_g = self.criterion_bce(global_pointer,
                                    data['selector_index'])
        loss_v = masked_cross_entropy(decoder_outputs_vocab.transpose(0, 1).contiguous(),
                                      data['sketch_response'].contiguous(),
                                      data['response_lengths'])
        loss_l = masked_cross_entropy(decoder_outputs_ptr.transpose(0, 1).contiguous(),
                                      data['ptr_index'].contiguous(),
                                      data['response_lengths'])

        loss = loss_g + loss_v + loss_l

        loss_dict = {"loss": loss,
                     "loss_g": loss_g,
                     "loss_v": loss_v,
                     "loss_l": loss_l
        }

        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=grad_clip)
            optimizer.step()

        return loss_dict

    def generate(self, batch_iter, output_dir, verbose=False):
        print("starting generation...")
        self.eval()
        
        refs, hyps, hyps_sketch = [], [], []

        for i, batch in enumerate(batch_iter):
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse, global_pointer = self.encode_and_decode(
                batch, self.max_resp_len,
                use_teacher_forcing=False,
                get_decoded_words=True
            )
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)

            for j, line in enumerate(decoded_fine):
                sent = []
                sent_coarse = []
                for w in line:
                    if w == 'EOS':
                        break
                    else:
                        sent.append(w)
                for wc in decoded_coarse[j]:
                    if wc == 'EOS':
                        break
                    else:
                        sent_coarse.append(wc)
                gold_sent = batch['response_plain'][j].strip().split(" ")
                refs.append(gold_sent)
                hyps.append(sent)
                hyps_sketch.append(sent_coarse)
                if verbose:
                    print("Gold:", " ".join(gold_sent))
                    print("Response:", " ".join(sent))
                    print("Response_sketch:", " ".join(sent_coarse))
                    print('\n')

        with open("%s/test.result.final" % output_dir, 'w') as fw:
            for i, res in enumerate(hyps):
                fw.write(" ".join(res))
                if i < len(hyps) - 1:
                    fw.write('\n')
        with open("%s/test.result.sketch" % output_dir, 'w') as fw:
            for i, res in enumerate(hyps_sketch):
                fw.write(" ".join(res))
                if i < len(hyps_sketch) - 1:
                    fw.write('\n')

        # show results
        sents = []
        for i, text in enumerate(refs):
            sents.append([hyps[i], refs[i]])

        # calc f1
        f1 = calc_f1(sents)
        # calc bleu
        bleu1, bleu2 = calc_bleu(sents)
        # calc distinct
        distinct1, distinct2 = calc_distinct(sents)

        output_str = "F1: %.2f%%\n" % (f1 * 100)
        output_str += "BLEU1: %.3f%%\n" % bleu1
        output_str += "BLEU2: %.3f%%\n" % bleu2
        output_str += "DISTINCT1: %.3f%%\n" % distinct1
        output_str += "DISTINCT2: %.3f%%\n" % distinct2

        print(output_str)
