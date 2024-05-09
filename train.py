import torch
from torch.utils.data import DataLoader

from dataset import CorpusPreprocess, TrainData
from model import GloVe
from tools import loss_func, save_word_vector


def train():
    use_gpu = torch.cuda.is_available()
    corpus_file_name = "data/zhihu.txt"
    save_vector_file_name = "data/glove.txt"
    epoches = 20
    min_count = 5 # min number of words per sentence
    batch_size = 512
    windows_size = 5
    vector_size = 300
    learning_rate = 0.001

    # get dataset
    print('load dataset ...')
    corpus_preprocessor = CorpusPreprocess(corpus_file_name, min_count)
    coo_matrix = corpus_preprocessor.get_cooccurrence_matrix(windows_size)
    vocab = corpus_preprocessor.get_vocab()
    print('done!')

    # gen dataset
    train_data = TrainData(coo_matrix)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # gen model
    model = GloVe(vocab, vector_size)
    if use_gpu:
        model.cuda()

    # gen optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    steps = 0
    for epoch in range(epoches):
        print(f"currently epoch is {epoch}, all epoch is {epoches}")
        avg_epoch_loss = 0
        for i, batch_data in enumerate(data_loader):
            c = batch_data['c']
            s = batch_data['s']
            X_c_s = batch_data['X_c_s']
            W_c_s = batch_data["W_c_s"]

            if use_gpu:
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            W_c_s_hat = model(c, s)
            loss = loss_func(W_c_s_hat, X_c_s, W_c_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_epoch_loss += loss / len(train_data)
            if steps % 1000 == 0:
                print(f"Steps {steps}, loss is {loss.item()}")
            steps += 1
        print(f"Epoches {epoch}, complete!, avg loss {avg_epoch_loss}.\n")

    # save model
    save_word_vector(save_vector_file_name, corpus_preprocessor, model, use_gpu)


if __name__ == "__main__":
    train()