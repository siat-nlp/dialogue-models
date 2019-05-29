import torch
from torch.utils.data import Dataset

UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3
TOPIC_A_PTR = 4
TOPIC_B_PTR = 5

MEM_TOKEN_SIZE = 4


class Lang:
    def __init__(self):
        self.vocab_count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
        self.n_words = len(self.index2word)  # Count default tokens

    def _id2word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def _count(self, word):
        if word in self.vocab_count.keys():
            self.vocab_count[word] += 1
        else:
            self.vocab_count[word] = 1

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self._id2word(word)
                self._count(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self._id2word(word)
                    self._count(word)


class LicDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, use_cuda=True):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.use_cuda = use_cuda

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        selector_index = torch.Tensor(self.data_info['selector_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if story_dim:
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i,:end,:] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        #data.sort(key=lambda x: len(x['conv_arr']), reverse=True)

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        selector_index, _ = merge_index(item_info['selector_index'])
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        # convert to contiguous and cuda
        if self.use_cuda:
            context_arr = context_arr.contiguous().cuda()
            response = response.contiguous().cuda()
            selector_index = selector_index.contiguous().cuda()
            ptr_index = ptr_index.contiguous().cuda()
            conv_arr = conv_arr.transpose(0, 1).contiguous().cuda()
            sketch_response = sketch_response.contiguous().cuda()
            if len(list(kb_arr.size())) > 1:
                kb_arr = kb_arr.transpose(0, 1).contiguous().cuda()

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        return data_info