import torch
from IPython.display import clear_output
from tqdm import tqdm

from lib.utils_nca import make_rand_sphere_masks
from lib.sample_pool import SamplePool


class Trainer:
    def __init__(self, params, model, loss_function, optimizer, scheduler=None):
        self.params = params
        self.loss_log = []
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

    def log_loss(self, loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("\n".join([ str(item) for item in self.loss_log ]))

    @staticmethod
    def batch_mse_loss(x, target):
        return ((x - target) ** 2).mean(dim=[1, 2, 3, 4])

    def step(self, sample, target):
        min_steps, max_steps = self.params.steps[0], self.params.steps[1]
        x = self.model(sample, steps=torch.randint(min_steps, max_steps, (1,)))
        loss = self.loss_function(x[:, :, :, :, :4], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return x, loss

    def train(self, device, seed, target, epochs, model_path, hparams_path=None, loss_log_path=None):

        params = self.params

        if hparams_path is not None: params.save(hparams_path)

        pool = SamplePool(x=seed.unsqueeze(0).repeat(params.pool_sz, 1, 1, 1, 1))
        sz = seed.shape[2]
        seed, target = seed.to(device), target.to(device)
        batch = None

        self.model = self.model.train().to(device)

        loss_acc, avg_loss, prec_avg_loss, loss_c = 0.0, 0.0, 0.0, 0

        for epoch in tqdm(range(epochs + 1)):

            if params.use_pattern_pool:
                batch = pool.sample(params.batch_sz)
                sample = batch.x.to(device)
                # computes mse loss of current batch wrt target
                # uses loss to sort indices in descending order
                loss_rank = torch.argsort(Trainer.batch_mse_loss(sample[..., :4], target), descending=True)
                sample = sample[loss_rank]
                sample[:1] = seed
                # replaces item with highest loss with seed to avoid CATASTROPHIC FORGETTING
                damage_masks = make_rand_sphere_masks(params.damage_n, sz).to(device)
                for d in range(params.damage_n):
                    sample[-d:] *= damage_masks[d]
            else:
                sample = seed.unsqueeze(0).repeat(params.batch_sz, 1, 1, 1, 1).to(device)

            x, loss = self.step(sample, target)

            if params.use_pattern_pool:
                batch.x[:] = x.detach().cpu()
                batch.commit()

            loss_item = loss.item()
            loss_acc += loss_item
            avg_loss = loss_acc / (epoch + 1)
            self.loss_log.append(loss_item)

            if epoch % 100 == 0:
                clear_output()
                print("Epoch %d: loss = %f\tavg loss = %f %s" %
                      (epoch, loss_item, avg_loss, "↑" if prec_avg_loss <= avg_loss else "↓"))
                prec_avg_loss = avg_loss
                # visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
                # plot_loss(loss_log)
                torch.save(self.model.state_dict(), model_path)

        if loss_log_path is not None: self.log_loss(loss_log_path)