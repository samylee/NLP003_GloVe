import torch
from tqdm import tqdm


def loss_func(X_c_s_hat, X_c_s, W_c_s):
    X_c_s = X_c_s.view(-1,1)
    W_c_s = W_c_s.view(-1,1)
    loss = torch.sum(W_c_s.mul((X_c_s_hat - torch.log(X_c_s))**2))
    return loss


def save_word_vector(file_name, corpus_preprocessor, model, use_gpu):
    with open(file_name, "w", encoding="utf-8") as f:
        if use_gpu:
            c_vector= model.c_weight.weight.data.cpu().numpy()
            s_vector= model.s_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector
        else:
            c_vector= model.c_weight.weight.data.numpy()
            s_vector= model.s_weight.weight.data.numpy()
            vector = c_vector + s_vector
        for i in tqdm(range(len(vector))):
            word = corpus_preprocessor.idex2word[i]
            s_vec = vector[i]
            s_vec = [str(s) for s in s_vec.tolist()]
            write_line = word + " " + " ".join(s_vec)+"\n"
            f.write(write_line)
        print("Glove vector save complete!")