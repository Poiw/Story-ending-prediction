import torch
import numpy as np 
from torchvision import transforms
import paramcfg

options = paramcfg.options

def train_collate_fn(batch):
    length = len(batch)
    data = torch.Tensor(5, length, options.dim, options.maxSentenceLength)
    labels = torch.Tensor(length)

    for i in range(len(batch)):
        for j in range(5):
            data[j][i] = torch.FloatTensor(batch[i][j])
        labels[i] = batch[i][5]
    
    return data, labels

def valid_collate_fn(batch):
    length = len(batch)
    input = torch.Tensor(4, length, options.dim, options.maxSentenceLength)
    results = torch.Tensor(2, length, options.dim, options.maxSentenceLength)
    labels = torch.Tensor(length)

    for i in range(len(batch)):
        for j in range(4):
            input[j][i] = torch.FloatTensor(batch[i][j])
        for j in range(2):
            results[j][i] = torch.FloatTensor(batch[i][j+4])
        labels[i] = batch[i][6]
    
    return input, results, labels

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, Sentences, embeddingModel, labels):
        self.data = Sentences
        self.model = embeddingModel
        # self.sentences = [[], [], [], [], []]
        # for story in Sentences:
        #     for i in range(len(story)):
        #         s = story[i].split()[:-1]
        #         vecSentence = np.zeros((options.dim, options.maxSentenceLength))

        #         length = len(s)
        #         assert(length <= options.maxSentenceLength)

        #         for j, word in enumerate(s):
        #             vecWord = embeddingModel[word]
        #             vecSentence[:, j] = vecWord
        #         self.sentences[i].append(vecSentence)
        self.labels = labels

    def __getitem__(self, index):

        story = self.data[index]

        sentences = []
        for i in range(len(story)):
            s = story[i].split()[:-1]
            vecSentence = np.zeros((options.dim, options.maxSentenceLength))

            length = len(s)
            assert(length <= options.maxSentenceLength)

            for j, word in enumerate(s):
                vecWord = self.model[word]
                vecSentence[:, j] = vecWord
            sentences.append(vecSentence)
       
        # return self.sentences[0][index], self.sentences[1][index], self.sentences[2][index], self.sentences[3][index], self.sentences[4][index], self.labels[index]
        return sentences[0], sentences[1], sentences[2], sentences[3], sentences[4], self.labels[index]


    def __len__(self):
        return len(self.labels)

class ValidDataSet(torch.utils.data.Dataset):
    def __init__(self, Sentences, embeddingModel, labels):
        self.data = Sentences
        self.model = embeddingModel
        # self.sentences = [[], [], [], []]
        # self.results = [[], []]
        # for story in Sentences:
        #     for i in range(4):
        #         s = story[i].split()[:-1]
        #         vecSentence = np.zeros((options.dim, options.maxSentenceLength))

        #         length = len(s)
        #         assert(length <= options.maxSentenceLength)

        #         for j, word in enumerate(s):
        #             vecWord = embeddingModel[word]
        #             vecSentence[:, j] = vecWord
        #         self.sentences[i].append(vecSentence)
        #     for i in range(2):
        #         s = story[i+4].split()[:-1]
        #         vecSentence = np.zeros((options.dim, options.maxSentenceLength))
                
        #         length = len(s)
        #         assert(length <= options.maxSentenceLength)
                
        #         for j, word in enumerate(s):
        #             vecWord = embeddingModel[word]
        #             vecSentence[:, j] = vecWord
        #         self.results[i].append(vecSentence)
        self.labels = labels

    def __getitem__(self, index):

        sentences = []
        results = []
        story = self.data[index]

        for i in range(4):
            s = story[i].split()[:-1]
            vecSentence = np.zeros((options.dim, options.maxSentenceLength))

            length = len(s)
            assert(length <= options.maxSentenceLength)

            for j, word in enumerate(s):
                vecWord = self.model[word]
                vecSentence[:, j] = vecWord
            sentences.append(vecSentence)

        for i in range(2):
            s = story[i+4].split()[:-1]
            vecSentence = np.zeros((options.dim, options.maxSentenceLength))
            
            length = len(s)
            assert(length <= options.maxSentenceLength)
            
            for j, word in enumerate(s):
                vecWord = self.model[word]
                vecSentence[:, j] = vecWord
            results.append(vecSentence)
       
        # return self.sentences[0][index], self.sentences[1][index], self.sentences[2][index], self.sentences[3][index], self.results[0][index], self.results[1][index], self.labels[index]
        return sentences[0], sentences[1], sentences[2], sentences[3], results[0], results[1], self.labels[index]

    def __len__(self):
        return len(self.labels)