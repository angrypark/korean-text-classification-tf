from collections import Counter, defaultdict
from queue import Queue

import tensorflow as tf
import numpy as np
import random
import sys
import threading


class Vectorizer:
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3
    pre_vocab = ['PAD', 'UNK', 'SOS', 'EOS']

    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.idx_to_vocab = self.pre_vocab \
                                + [line.strip().split('\t')[0] for line in f]
            self.vocab_to_idx = {v: i for i, v in enumerate(self.idx_to_vocab)}

    @property
    def num_vocabs(self):
        return len(self.idx_to_vocab)

    def vectorize(self, sentence):
        return [self.vocab_to_idx.get(token, self.UNK) for token in
                sentence.split()]

    def stringize(self, tokens, delimiter=' '):
        return delimiter.join([self.idx_to_vocab[idx] for idx in tokens])


class Dataset:
    def __init__(self,
                 vectorizer,
                 max_instance_num,
                 add_eos=False,
                 num_negative=0,
                 distortion=0.5,
                 use_class_weight=False,
                 classes=None,
                 use_class_bow=False,
                 use_dialogue_act=False,
                 use_emoji=False):
        if type(vectorizer) == str:
            self.vectorizer = Vectorizer(vectorizer)
        else:
            self.vectorizer = vectorizer

        if classes:
            if type(classes) == str:
                with open(classes, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f]
            self.idx_to_class = sorted(classes)
            self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}
        else:
            self.idx_to_class = []
            self.class_to_idx = {}
        self.class_bows, self.class_bow_weights, self.minimal_vocab_mapper = \
            self._build_class_vocab(self.idx_to_class)
        self.class_emojis = []

        self.class_weights = []  # weights to use for loss calculation
        self.class_probs = []    # original class probabilities
        self.distorted_class_probs = []  # distorted class probabilities
        self.train_data = {}

        self.val_data = []
        self.val_queries = []
        self.val_string_queries = []
        self.val_evaluation_data = []
        self.num_train_instances = 0     # number of train instances

        self.max_instance_num = max_instance_num
        self.add_eos = add_eos
        self.num_negative = num_negative
        self.distortion = distortion
        self.use_class_weight = use_class_weight
        self.use_class_bow = use_class_bow
        self.use_emoji = use_emoji

        self.use_dialogue_act = use_dialogue_act
        self.num_da_classes = 0
        self.dialogue_act_class_weights = {}

    @property
    def num_classes(self):
        return len(self.idx_to_class)

    def build(self, train_corpus_file, val_corpus_file,
              emoji_file=None, min_send_length=1, max_send_length=20,
              verbose=0):
        self.train_data = defaultdict(list)
        if self.use_dialogue_act:
            dialogue_act_counts = {}

        verbosity = verbose > 0
        if verbose == 0:
            verbose = 2 ** 30

        # build train corpus
        with open(train_corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                if idx % verbose == 0 and verbosity:
                    sys.stdout.write('\rProcessing... Line #{}'.format(idx))

                splits = line.strip().split('\t')  # send, recv, (da)
                tokens = self.vectorizer.vectorize(splits[0])
                if len(tokens) < min_send_length or len(
                        tokens) > max_send_length:
                    continue
                if self.add_eos:
                    tokens += [self.vectorizer.EOS]

                tokens = np.array(tokens, np.int32)
                if self.use_dialogue_act:
                    da = int(splits[2])
                    instance = (tokens, da)
                    self.num_da_classes = max(self.num_da_classes, da+1)
                else:
                    instance = (tokens, )
                self.train_data[splits[1]].append(instance)

            if verbosity:
                sys.stdout.write('\n')

        self.train_data = dict(self.train_data)
        self.idx_to_class = sorted(self.train_data.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}
        self.class_bows, self.class_bow_weights, self.minimal_vocab_mapper = \
            self._build_class_vocab(self.idx_to_class)

        # build val corpus
        val_labels = {}
        with open(val_corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                send, recv, score = line.strip().split('\t')
                tokens = self.vectorizer.vectorize(send)
                if self.add_eos:
                    tokens += [self.vectorizer.EOS]

                tup_tokens = tuple(tokens)
                label_list = val_labels.get(tup_tokens, list())
                label_list.append((self.class_to_idx[recv], float(score)))
                val_labels[tup_tokens] = label_list

        # val_labels: tokens -> (class_idxs, scores)
        for k, v in val_labels.items():
            sorted_v = sorted(v, key=lambda x: -x[1])
            idxs, scores = zip(*sorted_v)
            val_labels[k] = (np.array(idxs, np.int32),
                             np.array(scores, np.float32))

        if self.use_emoji and emoji_file is not None:
            emoji_mapper = {}
            with open(emoji_file) as f:
                for line in f:
                    class_string, emoji = line.strip().split('\t')
                    emoji_mapper[class_string] = emoji
            self.class_emojis = np.array(
                [emoji_mapper[c] for c in self.idx_to_class], dtype=np.int32)

        self.val_data = list()  # additional list for iteration
        self.val_queries = list()
        self.val_evaluation_data = list()
        for k, v in sorted(val_labels.items(), key=lambda x: x[0]):
            sentence_array = np.array(k)
            val_data = {
                'tokens': sentence_array,
                'class': v[0][0]  # just use first class
            }
            if self.use_dialogue_act:
                # fake da for running
                val_data['dialogue_act'] = 0

            self.val_data.append(val_data)
            self.val_queries.append(sentence_array)
            self.val_evaluation_data.append(v)
        self.val_string_queries = [self.vectorizer.stringize(q)
                                   for q in self.val_queries]

        num_classes = len(self.idx_to_class)
        num_class_instances = [
            min(len(self.train_data[self.idx_to_class[i]]),
                self.max_instance_num)
            for i in range(num_classes)]
        self.num_train_instances = sum(num_class_instances)

        if self.use_class_weight:
            self.class_weights = np.array(
                 [self.num_train_instances / (num_classes * nci) for nci in
                  num_class_instances])
        else:
            self.class_weights = np.ones([num_classes, ], dtype=np.float32)

        full_num_class_instances = [len(self.train_data[self.idx_to_class[i]])
                                    for i in range(num_classes)]
        full_num_instances = float(sum(full_num_class_instances))
        self.class_probs = np.array(
            [nci / full_num_instances for nci in full_num_class_instances])

        distorted_probs = []
        for i in range(self.num_classes):
            probs = np.copy(self.class_probs)
            probs[i] = 0.
            dist_probs = probs ** self.distortion
            dist_probs /= np.sum(dist_probs)
            distorted_probs.append(dist_probs)

        self.distorted_class_probs = np.array(distorted_probs)

        # build dialogue act class weight with sampled epoch
        if self.use_dialogue_act:
            sample_counter = Counter()
            total_counter = Counter()
            for instances in self.train_data.values():
                if len(instances) <= self.max_instance_num:
                    sampled_instances = instances
                else:
                    sampled_instances = random.sample(
                        instances, self.max_instance_num)
                sample_counter += Counter(inst[1] for inst in sampled_instances)
                total_counter += Counter(inst[1] for inst in instances)

            num_sampled_instances = sum(sample_counter.values())
            num_total_instances = sum(total_counter.values())
            sample_ratio = num_sampled_instances / num_total_instances

            self.dialogue_act_class_weights = {
                k: sample_ratio * total_count / sample_counter[k]
                for k, total_count in total_counter.items()
            }

            # debug
            from pprint import pprint
            pprint(self.dialogue_act_class_weights)

    def _build_class_vocab(self, classes):
        vocabs = [self.vectorizer.vectorize(cls) for cls in classes]
        max_length = max(len(v) for v in vocabs) if len(vocabs) > 0 else 0

        minimal_vocab_idxs = {self.vectorizer.PAD}
        for vs in vocabs:
            minimal_vocab_idxs.update(vs)
        minimal_vocab_idxs = sorted(list(minimal_vocab_idxs))
        minimal_vocab_idx_mapper = \
            {v: i for i, v in enumerate(minimal_vocab_idxs)}

        pad_idx = minimal_vocab_idx_mapper[self.vectorizer.PAD]
        tokens = np.ones((len(vocabs), max_length), dtype=np.int32) * pad_idx

        for i, token_arr in enumerate(vocabs):
            mapped_token_arr = \
                np.array([minimal_vocab_idx_mapper[idx] for idx in token_arr],
                         dtype=np.int32)
            tokens[i, :len(token_arr)] += mapped_token_arr

        weights = (tokens != minimal_vocab_idx_mapper[self.vectorizer.PAD])
        if self.vectorizer.UNK in minimal_vocab_idx_mapper:
            weights &= (tokens != minimal_vocab_idx_mapper[self.vectorizer.UNK])
        weights = weights.astype(np.float32)

        return tokens, weights, minimal_vocab_idx_mapper

    def _sample_negative(self, target, num):
        probs = self.distorted_class_probs[target]
        choices = np.random.choice(self.num_classes, num,
                                   replace=False, p=probs)
        return choices

    def _batchify(self, instances,
                  add_class_info,
                  num_negative,
                  add_class_bow_info,
                  add_dialogue_act_info,
                  add_emoji_info):
        lengths = np.array([len(inst['tokens']) for inst in instances],
                           dtype=np.int32)
        max_lengths = max(lengths)

        pad_idx = self.vectorizer.PAD
        if pad_idx == 0:
            tokens = np.zeros((len(instances), max_lengths), dtype=np.int32)
        else:
            tokens = np.ones((len(instances), max_lengths), dtype=np.int32)\
                     * pad_idx

        for i, inst in enumerate(instances):
            token_arr = inst['tokens']
            tokens[i, :len(token_arr)] += token_arr

        batch_dict = {
            'tokens': tokens,
            'lengths': lengths,
        }

        if add_class_info:
            classes = np.array([inst['class'] for inst in instances],
                               dtype=np.int32)
            class_weights = self.class_weights[classes]
            batch_dict.update({
                'classes': classes,
                'class_weights': class_weights
            })

            if num_negative > 0:
                if num_negative >= self.num_classes:
                    negative_classes = np.tile(
                        np.reshape(np.arange(self.num_classes, dtype=np.int32),
                                   [1, -1]),
                        (len(instances), 1)
                    )
                else:
                    negative_classes = np.array(
                        [self._sample_negative(cidx, num_negative)
                         for cidx in classes],
                        dtype=np.int32)
                batch_dict['negative_classes'] = negative_classes

            if add_class_bow_info:
                class_bows = [self.class_bows[cidx] for cidx in classes]
                class_bow_weights = \
                    [self.class_bow_weights[cidx] for cidx in classes]
                batch_dict['class_bows'] = class_bows
                batch_dict['class_bow_weights'] = class_bow_weights

            if add_dialogue_act_info:
                dialogue_acts = np.array(
                    [inst['dialogue_act'] for inst in instances],
                    dtype=np.int32)
                dialogue_act_weights = np.array(
                    [self.dialogue_act_class_weights[inst['dialogue_act']]
                     for inst in instances],
                    dtype=np.float32)
                batch_dict['dialogue_acts'] = dialogue_acts
                batch_dict['dialogue_act_weights'] = dialogue_act_weights

            if add_emoji_info:
                emojis = np.array(
                    [self.class_emojis[cidx] for cidx in classes],
                    dtype=np.int32)
                batch_dict['emojis'] = emojis

        return batch_dict

    def iterate_train(self, batch_size):
        """build train dataset for single epoch and iterate"""
        curr_dataset = []

        def _generate_instance(inst, cidx):
            instance = {'tokens': inst[0], 'class': cidx}
            if self.use_dialogue_act:
                instance['dialogue_act'] = inst[1]
            return instance

        for class_string, instances in self.train_data.items():
            if len(instances) <= self.max_instance_num:
                sampled_instances = instances
            else:
                sampled_instances = random.sample(instances,
                                                  self.max_instance_num)

            cidx = self.class_to_idx[class_string]
            curr_dataset.extend(
                [_generate_instance(inst, cidx) for inst in sampled_instances])

        random.shuffle(curr_dataset)
        # python sorts are stable!
        curr_dataset.sort(key=lambda x: len(x['tokens']))

        start_idxs = list(range(0, len(curr_dataset), batch_size))
        random.shuffle(start_idxs)
        for start in start_idxs:
            end = min(len(curr_dataset), start + batch_size)
            yield self._batchify(curr_dataset[start:end],
                                 add_class_info=True,
                                 num_negative=self.num_negative,
                                 add_class_bow_info=self.use_class_bow,
                                 add_dialogue_act_info=self.use_dialogue_act,
                                 add_emoji_info=self.use_emoji)

    def iterate_val(self, batch_size):
        for start in range(0, len(self.val_queries), batch_size):
            end = min(len(self.val_queries), start + batch_size)
            num_negative = self.num_classes if self.num_negative > 0 else 0
            yield self._batchify(self.val_data[start:end],
                                 add_class_info=True,
                                 num_negative=num_negative,
                                 add_class_bow_info=self.use_class_bow,
                                 add_dialogue_act_info=self.use_dialogue_act,
                                 add_emoji_info=self.use_emoji)

    def infer_batch(self, sentences):
        """
        Convert list of sentences into data dictionary for inference
        :param sentences: list of tokenized sentences
        :return: data dictionary
        """
        instances = []
        for sentence in sentences:
            tokens = self.vectorizer.vectorize(sentence)
            if self.add_eos:
                tokens += [self.vectorizer.EOS]
            instances.append({'tokens': tokens})

        num_negative = self.num_classes if self.num_negative > 0 else 0
        return self._batchify(instances,
                              add_class_info=False,
                              num_negative=num_negative,
                              add_class_bow_info=False,
                              add_dialogue_act_info=False,
                              add_emoji_info=False)

    def build_placeholder(self):
        placeholders = {
            'tokens': tf.placeholder(tf.int32, [None, None], 'tokens'),
            'lengths': tf.placeholder(tf.int32, [None], 'lengths'),
            'classes': tf.placeholder(tf.int32, [None], 'classes'),
            'class_weights': tf.placeholder(tf.float32, [None], 'class_weights')
        }

        if self.num_negative > 0:
            placeholders['negative_classes'] = \
                tf.placeholder(tf.int32, [None, None], 'negative_classes')
        if self.use_class_bow:
            max_bow_length = self.class_bows.shape[1]
            placeholders['class_bows'] = \
                tf.placeholder(tf.int32, [None, max_bow_length],
                               'class_bows')
            placeholders['class_bow_weights'] = \
                tf.placeholder(tf.float32, [None, max_bow_length],
                               'class_bow_weights')
        if self.use_dialogue_act:
            placeholders['dialogue_acts'] = \
                tf.placeholder(tf.int32, [None], 'dialogue_acts')
            placeholders['dialogue_act_weights'] = \
                tf.placeholder(tf.float32, [None], 'dialogue_act_weights')
        if self.use_emoji:
            placeholders['emojis'] = \
                tf.placeholder(tf.int32, [None], 'emojis')

        return placeholders

    @property
    def additional_hparams(self):
        additional_hparams = {
            'num_vocabs': self.vectorizer.num_vocabs,
            'num_classes': self.num_classes,
            'num_bow_vocabs': len(self.minimal_vocab_mapper)
        }
        if self.use_emoji:
            additional_hparams.update({
                'num_emojis': int(max(self.class_emojis) + 1)
            })
        if self.use_dialogue_act:
            additional_hparams.update({
                'num_da_classes': self.num_da_classes
            })
        return additional_hparams


def train_loop(dataset, batch_size, output_queue):
    for item in dataset.iterate_train(batch_size):
        output_queue.put(item, block=True)
    output_queue.put(None)


class ThreadingDataset:
    """
    Wrapper class of Dataset class that iterates train dataset in another thread
    """
    def __init__(self, dataset, max_queue_size=100):
        self._dataset = dataset
        self._max_queue_size = max_queue_size

    def __getattr__(self, key):
        try:
            return getattr(self._dataset, key)
        except AttributeError as e:
            raise e

    def _iterate(self, target_fn, batch_size, shuffle):
        q = Queue(maxsize=self._max_queue_size)
        t = threading.Thread(target=target_fn,
                             args=(self._dataset, batch_size, shuffle, q))
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()

    def iterate_train(self, batch_size):
        q = Queue(maxsize=self._max_queue_size)
        t = threading.Thread(target=train_loop,
                             args=(self._dataset, batch_size, q))
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()

    def iterate_val(self, batch_size):
        yield from self._dataset.iterate_val(batch_size)
