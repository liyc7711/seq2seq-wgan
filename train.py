import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.nn import functional as F
from models import Encoder, Decoder, Seq2Seq, Discriminator
from utils import load_dataset
from logger import VisdomWriter, log_samples


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-iterations', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-d_steps', type=int, default=5,
                   help='discriminator iterations')
    p.add_argument('-g_steps', type=int, default=1,
                   help='generator iterations')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


def penalize_grad(D, real, fake, batch_size, lamb):
    alpha = torch.rand(batch_size, 1, 1).expand(real.size()).cuda()
    interpolates = alpha * real + ((1 - alpha) * fake).cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    ones = torch.ones(d_interpolates.size()).cuda()
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=ones, create_graph=True,
                     retain_graph=True, only_inputs=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return grad_penalty


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        output = model(src, trg)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg.contiguous().view(-1), ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train_discriminator(D, G, optim_D, real_src, real_trg, lamb):
    D.zero_grad()
    batch_size = real_src.size(1)
    # train with real
    d_real = D(real_src, real_trg)
    d_real = d_real.mean()
    d_real.backward(mone)
    # train with fake
    fake_trg = G(real_src, real_trg)
    fake_trg = Variable(fake_trg.data)
    fake_trgv = fake_trg
    d_fake = D(real_src, fake_trgv)
    d_fake = d_fake.mean()
    d_fake.backward(one)
    # calculate gradient panalty
    grad_penalty = penalize_grad(D, real_src.data, fake_trg.data,
                                 batch_size, lamb)
    grad_penalty.backward()
    loss_d = d_fake - d_real + grad_penalty
    wasserstein = d_real - d_fake
    optim_D.step()
    return loss_d, wasserstein


def train_generator(D, G, optim_G, real_src, real_trg):
    G.zero_grad()
    fake_trg = G(real_src, real_trg)
    fake_trg = Variable(fake_trg.data)
    loss_g = D(fake_trg)
    loss_g = loss_g.mean()
    loss_g.backward(mone)
    loss_g = -loss_g
    optim_G.step()
    return loss_g


def eval_discriminator(D, G, optim_D, real_src, real_trg, lamb):
    D.zero_grad()
    batch_size = real_src.size(1)
    # train with real
    d_real = D(real_src, real_trg)
    d_real = d_real.mean()
    # train with fake
    fake_trg = G(real_src, real_trg)
    fake_trg = Variable(fake_trg.data)
    fake_trgv = fake_trg
    d_fake = D(real_src, fake_trgv)
    d_fake = d_fake.mean()
    # calculate gradient panalty
    grad_penalty = penalize_grad(D, real_src.data, fake_trg.data,
                                 batch_size, lamb)
    loss_d = d_fake - d_real + grad_penalty
    wasserstein = d_real - d_fake
    return loss_d, wasserstein


def eval_generator(D, G, optim_G, real_src, real_trg):
    fake_trg = G(real_src, real_trg)
    fake_trg = Variable(fake_trg.data)
    loss_g = D(fake_trg)
    loss_g = loss_g.mean()
    loss_g = -loss_g
    return loss_g


def main():
    args = parse_arguments()
    hidden_size = 1024
    embed_size = 512
    assert torch.cuda.is_available()

    # visdom for plotting
    vis_g = VisdomWriter("Training Generator Loss",
                         xlabel='Batch', ylabel='Loss')
    vis_d = VisdomWriter("Training Discriminator Loss",
                         xlabel='Batch', ylabel='Loss')
    vis_val = VisdomWriter("Validation Loss", xlabel='Batch', ylabel='Loss')

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    train_iter = iter(train_iter)
    val_iter = iter(val_iter)
    test_iter = iter(test_iter)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("de_vocab_size: %d en_vocab_size: %d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    G = Seq2Seq(encoder, decoder).cuda()
    D = Discriminator(en_size, embed_size, hidden_size, n_classes=2).cuda()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr)
    print(G)
    print(D)

    global one, mone
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()

    best_val_loss = None
    for i in range(1, args.iterations+1):
        # (1) Update D network
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = True
        for iter_d in range(args.d_steps):
            batch = next(train_iter)
            (src, len_src), (trg, len_trg) = batch.src, batch.trg
            real_src, real_trg = Variable(src.cuda()), Variable(trg.cuda())
            loss_d, wasserstein = train_discriminator(
                    D, G, optimizer_D, real_src, real_trg, args.lamb)
        # (2) Update G network
        for p in D.parameters():
            p.requires_grad = False  # to avoid computation
        for iter_d in range(args.g_steps):
            batch = next(train_iter)
            (src, len_src), (trg, len_trg) = batch.src, batch.trg
            real_src, real_trg = Variable(src.cuda()), Variable(trg.cuda())
            loss_g = train_generator(D, G, optimizer_G, real_src, real_trg)

        # plot losses
        vis_d.updatE(loss_d.data[0])
        vis_g.updatE(loss_g.data[0])

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(G.state_dict(), './.save/wseq2seq_g_%d.pt' % (i))
            torch.save(D.state_dict(), './.save/wseq2seq_d_%d.pt' % (i))
            best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
