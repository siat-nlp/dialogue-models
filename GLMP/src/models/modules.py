import torch
import torch.nn as nn
from src.inputters.dataset import SOS_token, PAD_token


class ContextRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1, use_cuda=False):
        super(ContextRNN, self).__init__()      
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=self.dropout if self.n_layers > 1 else 0, bidirectional=True)
        self.W = nn.Linear(2*self.hidden_size, self.hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if self.use_cuda:
            return torch.zeros(2, bsz, self.hidden_size).cuda()
        else:
            return torch.zeros(2, bsz, self.hidden_size)

    def forward(self, input_seqs, input_lengths, hidden=None, batch_first=False):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long()) 
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2) 
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))

        input_lengths = torch.Tensor(input_lengths)
        sorted_input_lengths, indices = torch.sort(input_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        if batch_first:
            embedded_inputs = embedded[indices]
        else:
            embedded_inputs = embedded[:, indices]

        embedded = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sorted_input_lengths, batch_first=batch_first)
        outputs, hidden = self.gru(embedded, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first)

        if batch_first:
            distorted_outputs = outputs[desorted_indices]
        else:
            distorted_outputs = outputs[:, desorted_indices]

        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(distorted_outputs)
        return outputs.transpose(0, 1), hidden


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hop = max_hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        for hop in range(self.max_hop+1):
            C = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.m_story = []

    def reset_memory(self):
        self.m_story = []

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi]+conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, dh_hidden, dh_outputs):
        # Forward multiple hop mechanism
        u = [dh_hidden.squeeze(0)]
        story_size = story.size()
        self.reset_memory()
        for hop in range(self.max_hop):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  #.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e

            # local memory embedding
            embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)

            embed_A = self.dropout_layer(embed_A)
            
            if len(list(u[-1].size())) == 1:
                u[-1] = u[-1].unsqueeze(0)  # used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A*u_temp, 2)
            prob_ = self.softmax(prob_logit)
            
            embed_C = self.C[hop+1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)

            # local memory embedding
            embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        # the global memory pointer is the memory distribution
        global_pointer = self.sigmoid(prob_logit)
        # the memory readout is used as the encoded KB information
        kb_readout = u[-1]

        return global_pointer, kb_readout

    def forward(self, query_vector, global_pointer, use_pointer=True):
        u = [query_vector]
        prob_soft, prob_logits = [], []
        for hop in range(self.max_hop):
            m_A = self.m_story[hop]
            if use_pointer:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)

            if len(list(u[-1].size())) == 1:
                u[-1] = u[-1].unsqueeze(0)  # used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A*u_temp, 2)
            prob_soft = self.softmax(prob_logits)

            m_C = self.m_story[hop+1]
            if use_pointer:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)

            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, vocab_size, shared_emb, index2word, embedding_dim, dropout,
                 n_layer=1, use_cuda=False, use_record=False):
        super(LocalMemoryDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = shared_emb
        self.index2word = index2word
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.n_layer = n_layer
        self.use_cuda = use_cuda
        self.use_record = use_record

        self.dropout_layer = nn.Dropout(dropout)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, n_layer,
                                 dropout=self.dropout if self.n_layer > 1 else 0)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        # scores = F.softmax(scores_, dim=1)
        return scores_
    
    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches,
                max_target_length, batch_size, global_pointer,
                use_teacher_forcing=False, get_decoded_words=False):
        # Initialize variables for vocab and pointer
        decoder_outputs_vocab = torch.zeros(max_target_length, batch_size, self.vocab_size)
        decoder_outputs_ptr = torch.zeros(max_target_length, batch_size, story_size[1])
        decoder_input = torch.LongTensor([SOS_token] * batch_size)
        memory_mask_for_step = torch.ones(story_size[0], story_size[1])
        if self.use_cuda:
            decoder_outputs_vocab = decoder_outputs_vocab.cuda()
            decoder_outputs_ptr = decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()
            memory_mask_for_step = memory_mask_for_step.cuda()
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        decoded_fine, decoded_coarse = [], []
        
        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.embedding(decoder_input)) # b * e
            if len(embed_q.size()) == 1:
                embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)

            # query the external knowledge using the hidden state of sketch RNN
            query_vector = hidden[0]
            prob_soft, prob_logits = extKnow(query_vector, global_pointer)
            decoder_outputs_ptr[t] = prob_logits

            p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
            decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()
            
            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []
                
                for bi in range(batch_size):
                    token = topvi[bi].item()
                    temp_c.append(self.index2word[token])
                    if '@' in self.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi]-1: 
                                cw = copy_list[bi][toppi[:, i][bi].item()]            
                                break
                        temp_f.append(cw)
                        
                        if self.use_record:
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return decoder_outputs_vocab, decoder_outputs_ptr, decoded_fine, decoded_coarse


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
