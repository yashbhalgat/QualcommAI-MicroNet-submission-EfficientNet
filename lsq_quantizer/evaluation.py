import argparse
import os
import torch
from utils.effnet import efficientnet_b0
from utils.data_loader import dataloader_imagenet
from helpers import load_checkpoint
from utils.utilities import get_constraint, eval_performance
from utils.add_lsqmodule import add_lsqmodule
from micronet_score import get_micronet_score


def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--data_root', default=None, type=str)

    parser.add_argument('--weight_bits', required=True,  type=int)
    parser.add_argument('--activation_bits', default=0, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--cem', default=False, action='store_true', help='use cem-based bit-widths')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    constr_activation = get_constraint(args.activation_bits, 'activation')

    net = efficientnet_b0(quan_first=True,
                  quan_last=True,
                  constr_activation=constr_activation,
                  preactivation=False,
                  bw_act=args.activation_bits)
    test_loader = dataloader_imagenet(args.data_root, split='test', batch_size=args.batch_size)
    add_lsqmodule(net, bit_width=args.weight_bits)

    if args.cem:
        ##### CEM vector for 1.5x_W7A7_CEM
        cem_input = [7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7,
                     6, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7, 4,
                     7, 6, 7, 5, 7, 7, 7, 7, 7, 5, 7, 7, 7, 5, 5, 7, 7, 7,
                     5, 6, 7, 7, 7, 6, 4, 7, 7, 6, 5, 4, 7, 6, 5, 5, 4, 7,
                     7, 6, 5, 4, 7, 7, 6, 5, 5, 3]

        strategy_path = "lsq_quantizer/cem_strategy_relaxed.txt"
        with open(strategy_path) as fp:
            strategy = fp.readlines()
        strategy = [x.strip().split(",") for x in strategy]

        strat = {}
        act_strat = {}
        for idx, width in enumerate(cem_input):
            weight_layer_name = strategy[idx][1]
            act_layer_name = strategy[idx][0]
            for name, module in net.named_modules():
                if name.startswith('module'):
                    name = name[7:]  # remove `module.`
                if name==weight_layer_name:
                    strat[name] = int(cem_input[idx])
                if name==act_layer_name:
                    act_strat[name] = int(cem_input[idx])

        add_lsqmodule(net, bit_width=args.weight_bits, strategy=strat)

        for name, module in net.named_modules():
            if name in act_strat:
                if "_in_act_quant" in name or "first_act" in name or "_head_act_quant0" in name or "_head_act_quant1" in name:
                    temp_constr_act = get_constraint(act_strat[name], 'weight') #symmetric
                else:
                    temp_constr_act = get_constraint(act_strat[name], 'activation') #asymmetric
                module.constraint = temp_constr_act

    name_weights_old = torch.load(args.model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    load_checkpoint(net, name_weights_new)

    score = get_micronet_score(net, args.weight_bits, args.activation_bits, weight_strategy=strat, activation_strategy=act_strat)
    
    criterion = torch.nn.CrossEntropyLoss()

    # Calculate accuracy
    net = net.cuda()

    quan_perf_epoch = eval_performance(net, test_loader, criterion)
    accuracy = quan_perf_epoch[1]

    print("Accuracy:", accuracy)
    print("Score:", score)

main()
