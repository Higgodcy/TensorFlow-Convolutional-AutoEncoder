import numpy as np


class Iterator_sample:
    def __init__(self, sample_list, batch_size, sample_size, label_size):
        self.cur_offset_index = 0
        self.sample_list = sample_list
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.label_size = label_size
        return

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return

    def shuffle(self):
        np.random.shuffle(self.sample_list)
        self.cur_offset_index = 0
        return

    def reset_cnt(self):
        self.cur_offset_index = 0
        return

    def next_batch(self):
        if self.cur_offset_index + self.batch_size <= len(self.sample_list):
            return_samples = self.sample_list[self.cur_offset_index:(self.cur_offset_index + self.batch_size),
                             0:self.sample_size]
            return_labels = self.sample_list[self.cur_offset_index:(self.cur_offset_index + self.batch_size),
                            self.sample_size:(self.sample_size + self.label_size)]
            self.cur_offset_index += self.batch_size
            return True, return_samples, return_labels

        self.cur_offset_index = 0
        return False, None, None

    def get_batch_times(self):
        return len(self.sample_list) // self.batch_size