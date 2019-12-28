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

def train_collate_fn_BE(batch):
    length = len(batch)

    maxbody_length = 0
    maxend_length = 0
    for i in range(length):
        maxbody_length = max(maxbody_length, batch[i][0].shape[1])
        maxend_length = max(maxend_length, batch[i][1].shape[1])

    body = torch.zeros(length, options.dim, maxbody_length)
    end = torch.zeros(length, options.dim, maxend_length)
    labels = torch.Tensor(length)

    for i in range(len(batch)):

        body[i,:,0:batch[i][0].shape[1]] = torch.FloatTensor(batch[i][0])
        end[i,:,0:batch[i][1].shape[1]] = torch.FloatTensor(batch[i][1])

        labels[i] = batch[i][2]
    
    return body, end, labels

def valid_collate_fn_BE(batch):
    length = len(batch)

    maxbody_length = 0
    maxend_length1 = 0
    maxend_length2 = 0
    for i in range(length):
        maxbody_length = max(maxbody_length, batch[i][0].shape[1])
        maxend_length1 = max(maxend_length1, batch[i][1].shape[1])
        maxend_length2 = max(maxend_length2, batch[i][2].shape[1])

    body = torch.zeros(length, options.dim, maxbody_length)
    end1 = torch.zeros(length, options.dim, maxend_length1)
    end2 = torch.zeros(length, options.dim, maxend_length2)
    labels = torch.Tensor(length)

    for i in range(len(batch)):

        body[i,:,0:batch[i][0].shape[1]] = torch.FloatTensor(batch[i][0])
        end1[i,:,0:batch[i][1].shape[1]] = torch.FloatTensor(batch[i][1])
        end2[i,:,0:batch[i][2].shape[1]] = torch.FloatTensor(batch[i][2])

        labels[i] = batch[i][3]
    
    return body, end1, end2, labels

class TrainDataSetBE(torch.utils.data.Dataset):
    def __init__(self, Sentences, embeddingModel, labels):
        self.data = Sentences
        self.model = embeddingModel
        self.labels = labels

    def __getitem__(self, index):

        story = self.data[index]

        body_length = 0
        for i in range(4):
            s = story[i].split()[:-1]
            body_length = body_length + len(s)
        
        s = story[4].split()[:-1]
        end_length = len(s)

        body_vec = np.zeros((options.dim, body_length))
        end_vec = np.zeros((options.dim, end_length))

        cnt = 0
        for i in range(4):
            s = story[i].split()[:-1]

            for j, word in enumerate(s):
                vecWord = self.model[word]
                body_vec[:, cnt+j] = vecWord
            cnt += len(s)

        s = story[4].split()[:-1]
        for j, word in enumerate(s):
            vecWord = self.model[word]
            end_vec[:, j] = vecWord
       
        return body_vec, end_vec, self.labels[index]


    def __len__(self):
        return len(self.labels)

class ValidDataSetBE(torch.utils.data.Dataset):
    def __init__(self, Sentences, embeddingModel, labels):
        self.data = Sentences
        self.model = embeddingModel
        self.labels = labels

    def __getitem__(self, index):

        story = self.data[index]

        body_length = 0
        for i in range(4):
            s = story[i].split()[:-1]
            body_length = body_length + len(s)
        
        s = story[4].split()[:-1]
        end_length1 = len(s)

        s = story[5].split()[:-1]
        end_length2 = len(s)

        body_vec = np.zeros((options.dim, body_length))
        end_vec1 = np.zeros((options.dim, end_length1))
        end_vec2 = np.zeros((options.dim, end_length2))

        cnt = 0
        for i in range(4):
            s = story[i].split()[:-1]

            for j, word in enumerate(s):
                vecWord = self.model[word]
                body_vec[:, cnt+j] = vecWord
            cnt += len(s)

        s = story[4].split()[:-1]
        for j, word in enumerate(s):
            vecWord = self.model[word]
            end_vec1[:, j] = vecWord

        s = story[5].split()[:-1]
        for j, word in enumerate(s):
            vecWord = self.model[word]
            end_vec2[:, j] = vecWord
       
        return body_vec, end_vec1, end_vec2, self.labels[index]

    def __len__(self):
        return len(self.labels)