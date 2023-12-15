import unittest
from unittest import TestCase

import torch
from sklearn import linear_model
import numpy as np

from utils import get_2class_mnist, visualize_result
from model import LogisticRegression as LR


import pytorch_influence_functions as ptif

from pytorch_influence_functions.influence_function import (
    grad_z,
    s_test,
)
from pytorch_influence_functions.calc_influence_function import (
    calc_influence_single,
)

EPOCH = 10
BATCH_SIZE = 100
NUM_A, NUM_B = 1, 7
TEST_INDEX = 5
WEIGHT_DECAY = 0.01
OUTPUT_DIR = 'result'
SAMPLE_NUM = 50 * 2
RECURSION_DEPTH = 1000
R = 10

class TestLeaveOneOut(TestCase):
    def test_leave_one_out(self):

        gpus = 1 if torch.cuda.is_available() else 0

        (x_train, y_train), (x_test, y_test) = get_2class_mnist(NUM_A, NUM_B)
        train_sample_num = len(x_train)

        class CreateData(torch.utils.data.Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                out_data = self.data[idx]
                out_label = self.targets[idx]

                return out_data, out_label
        
        train_data = CreateData(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

        # prepare sklearn model to train w
        C = 1.0 / (train_sample_num * WEIGHT_DECAY)
        sklearn_model = linear_model.LogisticRegression(C=C, solver='lbfgs', tol=1e-8, fit_intercept=False)

        # prepare pytorch model to compute influence function
        torch_model = LR(weight_decay=WEIGHT_DECAY)

        # train
        sklearn_model.fit(x_train, y_train.ravel())
        print('LBFGS training took %s iter.' % sklearn_model.n_iter_)

        # assign W into pytorch model
        w_opt = sklearn_model.coef_.ravel()
        with torch.no_grad():
            torch_model.w = torch.nn.Parameter(
                torch.tensor(w_opt, dtype=torch.float)
            )
        
        # calculate original loss
        x_test_input = torch.FloatTensor(x_test[TEST_INDEX: TEST_INDEX+1])
        y_test_input = torch.LongTensor(y_test[TEST_INDEX: TEST_INDEX+1])

        test_data = CreateData(x_test[TEST_INDEX: TEST_INDEX+1], y_test[TEST_INDEX: TEST_INDEX+1])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

        if gpus >= 0:
            torch_model = torch_model.cuda()
            x_test_input = x_test_input.cuda()
            y_test_input = y_test_input.cuda()

        test_loss_ori = torch_model.loss(torch_model(x_test_input), y_test_input, train=False).detach().cpu().numpy()

        # # get test loss gradient
        # test_grad = torch.autograd.grad(test_loss_ori, torch_model.w)

        # # get inverse hvp (s_test)
        # print('Calculating s_test ...')
        # s_test = s_test_sample(
        #         torch_model, x_test_input, y_test_input, train_loader, gpu=gpus, damp=0, scale=25, recursion_depth=RECURSION_DEPTH, r=R
        #     )[0].detach().cpu().numpy()
        # # s_test = torch_model.sess.run(torch_model.inverse_hessian, feed_dict={torch_model.x: x_train, torch_model.y: y_train}) @ test_grad

        # print(s_test)

        # get train loss gradient and estimate loss diff
        # loss_diff_approx = np.zeros(train_sample_num)
        # for i in range(train_sample_num):
        #     x_input = torch.FloatTensor(x_train[i])
        #     y_input = torch.LongTensor(y_train[i])

        #     if gpus >= 0:
        #         x_input = x_input.cuda()
        #         y_input = y_input.cuda()

        #     train_loss = torch_model.loss(torch_model(x_input), y_input)
        #     train_grad = torch.autograd.grad(train_loss, torch_model.w)[0].detach().cpu().numpy()

        #     loss_diff_approx[i] = np.asscalar(train_grad.T @ s_test) / train_sample_num
        #     if i % 100 == 0:
        #         print('[{}/{}] Estimated loss diff: {}'.format(i+1, train_sample_num, loss_diff_approx[i]))

        loss_diff_approx, _, _, _, = calc_influence_single(torch_model, train_loader, test_loader, test_id_num=0, gpu=1,
                                    recursion_depth=RECURSION_DEPTH, r=R)
        loss_diff_approx = - torch.FloatTensor(loss_diff_approx).cpu().numpy()

        # get high and low loss diff indice
        sorted_indice = np.argsort(loss_diff_approx)
        sample_indice = np.concatenate([sorted_indice[-int(SAMPLE_NUM/2):], sorted_indice[:int(SAMPLE_NUM/2)]])

        # calculate true loss diff
        loss_diff_true = np.zeros(SAMPLE_NUM)
        for i, index in zip(range(SAMPLE_NUM), sample_indice):
            print('[{}/{}]'.format(i+1, SAMPLE_NUM))

            # get minus one dataset
            x_train_minus_one = np.delete(x_train, index, axis=0)
            y_train_minus_one = np.delete(y_train, index, axis=0)

            # retrain
            C = 1.0 / ((train_sample_num - 1) * WEIGHT_DECAY)
            sklearn_model_minus_one = linear_model.LogisticRegression(C=C, fit_intercept=False, tol=1e-8, solver='lbfgs')
            sklearn_model_minus_one.fit(x_train_minus_one, y_train_minus_one.ravel())
            print('LBFGS training took {} iter.'.format(sklearn_model_minus_one.n_iter_))

            # assign w on tensorflow model
            w_retrain = sklearn_model_minus_one.coef_.T.ravel()
            with torch.no_grad():
                torch_model.w = torch.nn.Parameter(
                    torch.tensor(w_retrain, dtype=torch.float)
                )
            
            if gpus >= 0:
                torch_model = torch_model.cuda()

            # get retrain loss
            test_loss_retrain = torch_model.loss(torch_model(x_test_input), y_test_input, train=False).detach().cpu().numpy()

            # get true loss diff
            loss_diff_true[i] = test_loss_retrain - test_loss_ori

            print('Original loss       :{}'.format(test_loss_ori))
            print('Retrain loss        :{}'.format(test_loss_retrain))
            print('True loss diff      :{}'.format(loss_diff_true[i]))
            print('Estimated loss diff :{}'.format(loss_diff_approx[index]))

        r2_score = visualize_result(loss_diff_true, loss_diff_approx[sample_indice])

        self.assertTrue(r2_score > 0.9)


if __name__ == "__main__":
    unittest.main()
