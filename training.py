from tqdm import tqdm
from loss import *
from visualizer import visualize_tensor_images


def train_cgan(
        gen,
        disc,
        gen_opt,
        disc_opt,
        dataloader,
        criterion,
        display_step,
        z_dim,
        n_epochs,
        viz,
        device='cpu'):

    cur_step = 0
    mean_gen_loss = 0
    mean_disc_loss = 0

    for epoch in range(n_epochs):
        for rimg, rlabel in tqdm(dataloader):
            cur_batch_size = len(rimg)
            rimg = rimg.to(device)
            disc_opt.zero_grad()
            disc_loss_val = disc_loss(
                gen,
                disc,
                criterion,
                rimg,
                rlabel,
                cur_batch_size,
                z_dim,
                device)
            disc_loss_val.backward(retain_graph=True)
            disc_opt.step()
            gen_opt.zero_grad()
            gen_loss_val = gen_loss(
                gen,
                disc,
                criterion,
                rlabel,
                cur_batch_size,
                z_dim,
                device)
            gen_loss_val.backward()
            gen_opt.step()
            mean_gen_loss += gen_loss_val.item() / display_step
            mean_disc_loss += disc_loss_val.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    "Epoch: {}/{}".format(epoch, n_epochs),
                    "Gen_loss: {:.4f}".format(mean_gen_loss),
                    "Disc_loss: {:.4f}".format(mean_disc_loss))
                if viz == True:
                    fake_noise = torch.randn(cur_batch_size, z_dim).to(device)
                    fake_images = gen(fake_noise,rlabel)
                    visualize_tensor_images(fake_images,size=(fake_images.shape[1],fake_images.shape[2],fake_images.shape[3]))
                    visualize_tensor_images(rimg,size=(rimg.shape[1],rimg.shape[2],rimg.shape[3]))
                mean_gen_loss = 0
                mean_disc_loss = 0

            cur_step += 1


