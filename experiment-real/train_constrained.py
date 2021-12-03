# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse, os
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='real', type=str, help='name of the task')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--dual_lr', default=0.00003, type=float, help='Dual LR')
    parser.add_argument('--primal_steps', default=50, type=int, help='Primal Steps')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Hamiltonian Derivative constraint')
    
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training Constrained")
  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # arrange data
  data = get_dataset('pend-real', args.save_dir)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  print(x.size())
  assert(0)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])
  mu = torch.zeros(1, dtype=torch.float32)
  epsilon = args.epsilon
  lr_dual = args.dual_lr
  primal_steps = args.primal_steps

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': [], 'mu': [], 'lagrangian': []}
  for step in range(args.total_steps+1):

    # train step
    optim.zero_grad()
    dxdt_hat = model.rk4_time_derivative(x, dt=1/6.) if args.use_rk4 else model.time_derivative(x)
    optim.zero_grad()
    loss = L2_loss(dxdt, dxdt_hat)
    slack = (dxdt_hat[:,0]+dxdt_hat[:,1]).mean()-epsilon
    lagrangian = loss + mu*slack
    lagrangian.backward()
    optim.step()
    optim.zero_grad()
    # Update dual variables
    if step%primal_steps ==0:
      with torch.no_grad():
        mu = torch.clamp(mu + lr_dual*(slack), min=0, max=None)

    # run validation
    test_dxdt_hat = model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)

    # logging
    stats['mu'].append(mu.item())
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}, mu {:.4e}, lagrangian  {:.4e}, slack {:.4e}".format(step, loss.item(), test_loss.item(), mu.item(), lagrangian.item(), slack.item()))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-constrained'
    label = '-baseline' + label if args.baseline else label
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)