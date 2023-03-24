# coding=utf-8
import os
import argparse
import json
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from dataset_pytorch import Dataset as EgoVQA_Dataset
from dataset_pororo import Dataset as Pororo_Dataset
from embed_loss import MultipleChoiceLoss
from attention_module import *
from util import AverageMeter

def collate_fn(data):
    questions_encoded = []
    labels = []
    answers_encoded = {'a1': [], 'a2': [], 'a3': [], 'a4': [], 'a5': []}
    vgg, c3d = [], []
    questions_length = []
    max_sequence_length = 30
    video_lengths = []
    question_words = []
    max_enc_len = 0
    answers_lengths = []
    qtexts = []
    atexts = {'a1': [], 'a2': [], 'a3': [], 'a4': [], 'a5': []}

    for i in range(len(data)):
        max_enc_len = 0
        current_question = torch.from_numpy(np.array(data[i][0].split(',')).astype(np.int64))
        questions_encoded.append(current_question)
        labels.append(torch.from_numpy(np.array(data[i][1])))

        #for value in range(2,7):
        #    answers_encoded.append(torch.from_numpy(np.array(data[i][value].split(',')).astype(np.int64)))

        answers = []
        answers_length = []
        for key, value in zip(answers_encoded, [x + 2 for x in range(6)]):
            current_answer = np.array(data[i][value].split(',')).astype(np.int64)
            max_enc_len = max(max_enc_len, len(current_answer))
            answers_encoded[key].append(torch.from_numpy(current_answer))
            answers.append(current_answer)
            answers_length.append(len(current_answer))

        answers_lengths.append(answers_length)

        question_words_tmp = []
        for j, answer in enumerate(answers):
            current_answer = torch.from_numpy(np.pad(answer, (0, max_enc_len-len(answer)), 'constant'))
            question_words_tmp.append(torch.cat((current_question, current_answer), 0))

        question_words.append(torch.stack(question_words_tmp))

        max_enc_len += len(current_question)

        vgg_feat, c3d_feat = data[i][7], data[i][8]
        vid_len = vgg_feat.shape[0]
        # print vgg_feat.shape, c3d_feat.shape, '===',
        if vid_len >= max_sequence_length * 2:
            ss = vid_len // max_sequence_length
            vgg_feat = vgg_feat[::ss, :]
            c3d_feat = c3d_feat[::ss, :]
        elif vid_len > max_sequence_length:
            ss = random.randint(0, vid_len - max_sequence_length - 1)
            vgg_feat = vgg_feat[ss:ss + max_sequence_length, :]
            c3d_feat = c3d_feat[ss:ss + max_sequence_length, :]

        video_lengths.append(torch.tensor(vgg_feat.shape[0]))
        vgg.append(vgg_feat)
        c3d.append(c3d_feat)
        questions_length.append(torch.tensor(max_enc_len))

        qtexts.append(data[i][12]["qtext"])
        for _k in ["a1", "a2", "a3", "a4", "a5"]:
            atexts[_k].append(data[i][12][_k])

    answers_lengths = torch.tensor(answers_lengths)
    return questions_encoded, labels, answers_encoded, vgg, c3d, questions_length, video_lengths, question_words, answers_lengths, qtexts, atexts


def main(args, seed):
    """Main script."""
    training_generator = []
    validation_generator = []
    test_generator = []

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)

    args.word_dim = 300
    args.vocab_num = 4000 if args.dataset == "egovqa" else 6833
    path_word_embedding = os.path.join('.',
                                       ('data' if args.dataset == "egovqa" else "data_pororo"),
                                       'word_embedding.npy')
    args.pretrained_embedding = path_word_embedding
    args.video_feature_dim = 4096
    args.video_feature_num = 20
    args.memory_dim = 256
    args.reg_coeff = 1e-5
    args.preprocess_dir = 'data'
    path_logs = os.path.join('.', 'logs')
    args.log = path_logs

    args.activation_function = nn.Sigmoid
    if (args.activation == 1):
        args.activation_function = nn.Tanh
    elif (args.activation == 2):
        args.activation_function = nn.ReLU
    elif (args.activation != 0):
        assert False
    name_activation_function = str(args.activation_function).split('.')[-1].split('\'')[0]

    args.name_model = 'baseline'

    comment = '_%s_s%d_memtype%s_%s' % (args.dataset, args.split, args.memory_type, args.embed_tech)
    if "ft" in args.embed_tech or "convLSTM" in args.embed_tech:
        comment = comment + '_embedlr%s' % (str(args.ft_lr))
    if args.pretrain == 'y':
        comment = comment + '_withpretrain'
    comment = comment + '_%s_batchsize_%s_lr_seed%d' % (str(args.batch_size), str(args.learning_rate), seed)
    if args.additional_tasks != "":
        comment = comment + "_tasks={}".format(args.additional_tasks)
    if args.use_int_mem:
        comment = comment + "_useintmem"
    if args.QT_bias:
        comment = comment + "_QTbias"
    if args.QT_weight:
        comment = comment + "_QTWiiWif"

    comment = comment + '_' + str(args.hidden_size) + '_hiddensize'

    writer = SummaryWriter(comment=comment)

    if args.dataset == "egovqa":
        video_cam = ['1_D', '1_M', '2_F', '2_M', '3_F', '3_M', '4_F', '4_M',
                     '5_D', '5_M', '6_D', '6_M', '7_D', '7_M', '8_M', '8_X']
        path_data_split = os.path.join('.', 'data', 'data_split.json')
        with open(path_data_split, 'r') as f:
            splits = json.load(f)
        assert args.split < len(splits)
        video_cam_split = splits[args.split]

        path_data = os.path.join('.', 'data')
        path_data_feats = os.path.join(path_data, 'feats')
    else:
        video_names = list()

        for l in json.load(open("data_pororo/pororo_qa.json"))["PororoQA"]:
            if l["video_name"] not in video_names:
                video_names.append(l["video_name"])
        video_names = sorted(video_names, key=lambda n: int(n.split("_ep")[1].replace(",", "")))  # list(sorted(video_names))
        video_cam_split = [video_names[:94], video_names[94:137],
                           video_names[137:]]
        path_data = 'data_pororo'
        path_data_feats = os.path.join(path_data, 'feats_pororo')

    batch_size = args.batch_size  #
    shuffle_option = True 
    num_workers = 0 

    params_training = {'batch_size': batch_size,
                        'shuffle': shuffle_option,
                        'num_workers': num_workers,
                        'collate_fn': collate_fn,
                        'drop_last': True}

    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': num_workers,
              'collate_fn': collate_fn}

    if args.dataset == "egovqa":
        _path = os.path.join(path_data, 'qa_data.json')
        total_qa = pd.read_json(_path)
    else:
        _path = os.path.join(path_data, 'pororo_qa.json')
        total_qa = json.load(open(_path))["PororoQA"]

    partition = {'train': [], 'validation': [], 'test': []}

    if args.dataset == "egovqa":
        for ix, data_group in enumerate(video_cam_split):
            # labels = labels + data_group
            if ix == 0:
                partition['train'] = total_qa[total_qa['video_cam'].isin(data_group)]
                partition['train'].reset_index(inplace=True)
            elif ix == 1:
                partition['validation'] = total_qa[total_qa['video_cam'].isin(data_group)]
                partition['validation'].reset_index(inplace=True)
            else:
                partition['test'] = total_qa[total_qa['video_cam'].isin(data_group)]
                partition['test'].reset_index(inplace=True)
    else:
        for _l in total_qa:
            if _l["video_name"] in video_cam_split[0]:
                partition["train"].append(_l)
            elif _l["video_name"] in video_cam_split[1]:
                partition["validation"].append(_l)
            elif _l["video_name"] in video_cam_split[2]:
                partition["test"].append(_l)
            else:
                assert False, "{} not found in video_cam_split".format(_l["video_name"])

        partition["train"] = pd.DataFrame(partition["train"])
        partition["validation"] = pd.DataFrame(partition["validation"])
        partition["test"] = pd.DataFrame(partition["test"])

    if args.dataset == "egovqa":
        dts = EgoVQA_Dataset
    else:
        dts = Pororo_Dataset

    use_precomp_fts = True
    if args.memory_type in ["_mrm2s_convLSTM", "_mrm2s_FTFCs"]:
        use_precomp_fts = False

    training_set = dts(partition['train'], video_cam_split[0], path_data_feats,
                           aug_tech=args.aug_tech, app_ft_vcs=use_precomp_fts, mot_ft_vcs=use_precomp_fts)
    training_generator = torch.utils.data.DataLoader(training_set, **params_training)

    validation_set = dts(partition['validation'], video_cam_split[1], path_data_feats,
                             app_ft_vcs=use_precomp_fts, mot_ft_vcs=use_precomp_fts)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    test_set = dts(partition['test'], video_cam_split[2], path_data_feats,
                       app_ft_vcs=use_precomp_fts, mot_ft_vcs=use_precomp_fts)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    args.image_feature_net = 'concat'
    args.layer = 'fc'

    if args.pretrain == 'n':
        args.save_model_path = os.path.join(args.save_path, 'model_{}{}_{}{}_{}_bs{}_lr{}{}_seed{}{}{}{}{}{}'.format(
            "" if args.dataset == "egovqa" else "pororo",
            args.image_feature_net,
            args.layer,
            args.memory_type,
            args.embed_tech,
            args.batch_size,
            args.learning_rate,
            "" if ("ft" not in args.embed_tech and "convLSTM" not in args.embed_tech) else "_embedlr{}".format(args.ft_lr),
            seed,
            "" if args.aug_tech == -1 else "_augtech{}".format(args.aug_tech),
            "" if args.additional_tasks == "" else "_tasks={}".format(args.additional_tasks),
            "" if not args.use_int_mem else "_useintmem",
            "" if not args.QT_bias else "_QTbias",
            "" if not args.QT_weight else "_QTWiiWif"
        ))
        #args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net, args.layer, args.memory_type)
    else:
        assert False, "args.pretrain == y, need to implement"
        args.save_model_path = os.path.join(args.save_path,
                                            'model_%s_%s_withpretrain_%s_%s_bs%s_lr%s' % (args.image_feature_net, args.layer,
                                                                                          args.memory_type, args.embed_tech,
                                                                                          (str(args.batch_size)), str(args.learning_rate)))
        # args.save_model_path = args.save_path + 'model_%s_%s_withpretrain_%s' % (
        # args.image_feature_net, args.layer, args.memory_type)

    args.save_model_path = os.path.join(args.save_model_path, '%s_hidden_size%s' % (name_activation_function, str(args.hidden_size)))

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(os.path.join(args.save_model_path, 's%d' % (args.split))):
        os.makedirs(os.path.join(args.save_model_path, 's%d' % (args.split)))

    args.pretrain_model_path = os.path.join('pretrain_models', 'model_%s_%s%s' % (args.image_feature_net, args.layer, args.memory_type))

    #############################
    # get video feature dimension
    #############################
    feat_channel = args.video_feature_dim
    feat_dim = 1
    if 'BERT' in args.memory_type or args.embed_tech in ["bert", "bert_ft", "bert_jointft"]:
        text_embed_size = 768  # args.word_dim per BERT-base
    elif 'XLM' in args.memory_type or args.embed_tech in ["xlm", "xlm_ft"]:
        text_embed_size = 2048
    elif 'ELMo' in args.memory_type or args.embed_tech in ["elmo", "elmo_ft"]:
        text_embed_size = 1024
    else:
        text_embed_size = 300

    voc_len = args.vocab_num
    num_layers = 2
    max_sequence_length = args.video_feature_num
    word_matrix = np.load(args.pretrained_embedding)

    activation_function = args.activation_function

    if args.memory_type in ['_stvqa', '_enc_dec']:
        feat_channel *= 2

    if args.QT_bias:
        net_build = QTBiasNetwork
        kwargs = {"qt_weighting": args.QT_weight}
    else:
        net_build = TheNetwork
        kwargs = {}

    rnn = net_build(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                     voc_len, num_layers, word_matrix, activation_function,
                     max_len=max_sequence_length, embed_tech=args.embed_tech,
                     additional_tasks=args.additional_tasks, architecture=args.memory_type,
                     use_int_mem=args.use_int_mem, device=device, dataset=args.dataset,
                     **kwargs)
    rnn = rnn.to(device)

    print(rnn)
    #################################
    # load pretrain model to finetune
    #################################
    if "_ft" in args.embed_tech:
        assert args.embed_tech in ["bert_ft", "elmo_ft", "xlm_ft", "glove_ft"]
        frozen_chkpts = args.save_model_path.replace("_ft", "" if args.embed_tech != "glove_ft" else "_frozen")
        frozen_chkpts = frozen_chkpts.replace("_embedlr{}".format(args.ft_lr), "")
        frozen_chkpts = os.path.join(frozen_chkpts, "s{}".format(args.split))
        print("looking for checkpoint in ", frozen_chkpts)
        files = sorted(os.listdir(frozen_chkpts))  #[:20]  # protocol 1: only first 20 models
        prefix = 'rnn-'
        max_val = 0
        max_test = 0
        max_iter = -1
        best_state = None
        for f in files:
            if f.startswith(prefix):
                segs = f.split('-')
                it0 = int(segs[1])
                val_acc = float(segs[2][3:])
                test_acc = float(segs[3][2:-4])
                if val_acc == max_val and test_acc > max_test:
                    max_iter = it0
                    max_test = test_acc
                    best_state = f
                elif val_acc > max_val:
                    max_iter = it0
                    max_val = val_acc
                    max_test = test_acc
                    best_state = f
        print('load checkpoint', best_state, max_val, max_test)
        assert best_state is not None
        rnn.load_state_dict(torch.load(os.path.join(frozen_chkpts, best_state), map_location=args.device))

    # loss function
    criterion = MultipleChoiceLoss(margin=1, size_average=True).to(device)
    if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(","):
        qt_classes_criterion = nn.BCELoss().to(device)

    if "jointft" in args.embed_tech:
        embed_lr = args.ft_lr
        _lr = args.learning_rate
        _params = [{"params": [p for (n, p) in rnn.named_parameters() if "embed" not in n]},
                   {"params": [p for (n, p) in rnn.named_parameters() if "embed" in n],
                    "lr": embed_lr}]
        optimizer = torch.optim.Adam(_params, lr=_lr, weight_decay=0.0005)
    elif "convLSTM" in args.memory_type and args.ft_lr is not None:
        embed_lr = args.ft_lr
        _lr = args.learning_rate
        _params = [{"params": [p for (n, p) in rnn.named_parameters() if "convLSTM" not in n]},
                   {"params": [p for (n, p) in rnn.named_parameters() if "convLSTM" in n],
                    "lr": embed_lr}]
        optimizer = torch.optim.Adam(_params, lr=_lr, weight_decay=0.0005)
    else:
        _lr = args.learning_rate if "_ft" not in args.embed_tech else args.ft_lr
        optimizer = torch.optim.Adam(rnn.parameters(), lr=_lr, weight_decay=0.0005)

    best_test_acc = 0.0
    best_test_iter = 0

    iter = 0

    for epoch in range(0, args.epoch):
        if "_ft" in args.embed_tech:
            assert args.embed_tech in ["bert_ft", "elmo_ft", "xlm_ft", "glove_ft"]
            for name, par in rnn.named_parameters():
                if "embed" not in name:
                    par.requires_grad_(False)

        batch_n = 0
        acc_cum = 0
        loss_cum = 0
        qt_classes_loss_cum = 0

        for current_batch in training_generator:
            for ix, dataset_item in enumerate(current_batch):
                if ix > 8:  # qtexts, atexts
                    break
                # for tensor_element in dataset_item:
                #     tensor_element.to(device)
                if ix == 2:  # to access the dictionary containing the possible answers
                    for answer in dataset_item:
                        for i, answer_item in enumerate(dataset_item[answer]):
                            dataset_item[answer][i] = answer_item.to(device)
                            # print(answer_item)
                elif ix == 8: # to access the answers_lengths
                    dataset_item = dataset_item.to(device)
                else:
                    for i, tensor_element in enumerate(dataset_item):
                        dataset_item[i] = tensor_element.to(device)
                        # print(tensor_element)
            iter += 1
            batch_n += 1
            questions_encoded, labels, answers_encoded, vgg, c3d, questions_length, video_lengths, question_words, answers_lengths, qtexts, atexts = current_batch
            answers_lengths = answers_lengths.to(device)

            data_dict = {}
            data_dict['video_features'] = [vgg, c3d]
            data_dict['video_lengths'] = video_lengths
            data_dict['question_words'] = question_words
            data_dict['answers'] = labels
            data_dict['question_lengths'] = questions_length
            data_dict['num_mult_choices'] = 5
            data_dict['answers_lengths'] = answers_lengths
            data_dict['qtexts'] = qtexts
            data_dict['atexts'] = atexts

            #writer.add_graph(rnn, data_dict)
            model_output = rnn(data_dict)
            """import torchviz
            dot = torchviz.make_dot(model_output["outputs"], params=dict(rnn.named_parameters()))
            # dot = torchviz.make_dot(model_output["qt_classes"], params=dict(rnn.named_parameters()))
            dot.render()
            exit(0)"""

            outputs = model_output["outputs"]
            predictions = model_output["predictions"]
            targets = data_dict['answers']

            loss = criterion(outputs, targets, device)
            loss_item = loss.item()

            if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",") and "_ft" not in args.embed_tech:
                qt_classes = model_output["qt_classes"]
                if args.dataset == "egovqa":
                    from dataset_pytorch import _qtypes_map, get_type
                else:
                    from dataset_pororo import _qtypes_map, get_type
                qt_classes_groundtruth = []
                for _bi, q in enumerate(data_dict['qtexts']):
                    for _c in range(5):
                        if args.dataset == "egovqa":
                            _ohe_idx = _qtypes_map[get_type(q)]
                        else:
                            a1 = data_dict["atexts"]["a1"][_bi]
                            a2 = data_dict["atexts"]["a2"][_bi]
                            a3 = data_dict["atexts"]["a3"][_bi]
                            a4 = data_dict["atexts"]["a4"][_bi]
                            a5 = data_dict["atexts"]["a5"][_bi]
                            _ohe_idx = _qtypes_map[get_type(q, a1, a2, a3, a4, a5)]
                        _qt_gt = torch.zeros(len(_qtypes_map))
                        _qt_gt = _qt_gt.scatter_(0, torch.tensor(_ohe_idx), 1)
                        #_qt_gt = torch.nn.functional.one_hot(torch.tensor(_ohe_idx))
                        _qt_gt = _qt_gt.to(device)
                        qt_classes_groundtruth.append(_qt_gt)
                qt_classes_groundtruth = torch.stack(qt_classes_groundtruth)
                qt_classes_loss = qt_classes_criterion(qt_classes.squeeze(0), qt_classes_groundtruth)
                loss = loss + qt_classes_loss  #.backward()
                #writer.add_scalar('Train/QT_classifier_loss (per batch)', qt_classes_loss.item(), iter)
                qt_classes_loss_cum += qt_classes_loss.item()

            if args.additional_tasks != "" and "qtclassif_unsup" in args.additional_tasks.split(","):
                print(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_size == 1:
                import torchviz
                print("Computing and rendering the computational graph")
                dot = torchviz.make_dot(loss, params=dict(rnn.named_parameters()))
                dot.render()
                exit(0)

            acc = rnn.accuracy(predictions, torch.stack(targets))
            print('Train iter %d, loss %.3f, acc %.2f' % (iter, loss.data, acc.item()))

            acc_cum += acc.item()
            loss_cum += loss_item

        writer.add_scalar('Train/loss (avg epoch)', loss_cum / batch_n, epoch)
        writer.add_scalar('Train/accuracy (avg epoch)', acc_cum / batch_n, epoch)
        if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",") and "_ft" not in args.embed_tech:
            writer.add_scalar('Train/QT_classifier_loss (avg epoch)', qt_classes_loss_cum / batch_n, epoch)

        if epoch >= 0:
            rnn.eval()

            # val iterate over examples
            with torch.no_grad():

                idx = 0
                accuracy = AverageMeter()

                for current_batch in validation_generator:
                    for ix, dataset_item in enumerate(current_batch):
                    #     for tensor_element in dataset_item:
                    #         tensor_element.to(device)
                        if ix > 8:
                            break
                        if ix == 2:  # to access the dictionary containing the possible answers
                            for answer in dataset_item:
                                for i, answer_item in enumerate(dataset_item[answer]):
                                    dataset_item[answer][i] = answer_item.to(device)
                                    # print(answer_item)
                        elif ix == 8:  # to access the answers_lengths
                            dataset_item = dataset_item.to(device)
                        else:
                            for i, tensor_element in enumerate(dataset_item):
                                dataset_item[i] = tensor_element.to(device)
                                # print(tensor_element)

                    if idx % 10 == 0:
                        print('Val iter %d/%d' % (idx, len(validation_generator)))
                    idx += 1

                    iter += 1
                    batch_n += 1
                    questions_encoded, labels, answers_encoded, vgg, c3d, questions_length, video_lengths, question_words, answers_lengths, qtexts, atexts = current_batch
                    answers_lengths = answers_lengths.to(device)

                    len_data = len(vgg)

                    data_dict = {}
                    data_dict['video_features'] = [vgg, c3d]
                    data_dict['video_lengths'] = video_lengths
                    data_dict['question_words'] = question_words
                    data_dict['answers'] = labels
                    data_dict['question_lengths'] = questions_length
                    data_dict['num_mult_choices'] = 5
                    data_dict['answers_lengths'] = answers_lengths
                    data_dict['qtexts'] = qtexts
                    data_dict['atexts'] = atexts

                    model_output = rnn(data_dict)
                    outputs = model_output["outputs"]
                    predictions = model_output["predictions"]
                    targets = data_dict['answers']

                    acc = rnn.accuracy(predictions, torch.stack(targets))
                    accuracy.update(acc.item(), len_data)

                    #writer.add_scalar('Val/accuracy (per example)', acc, idx)


                val_acc = accuracy.avg
                print('Val iter %d, acc %.3f' % (iter, val_acc))

                writer.add_scalar('Val/accuracy (avg)', val_acc, iter)

                idx = 0
                accuracy = AverageMeter()

                for current_batch in test_generator:
                    for ix, dataset_item in enumerate(current_batch):
                        if ix > 8:
                            break
                        if ix == 2:  # to access the dictionary containing the possible answers
                            for answer in dataset_item:
                                for i, answer_item in enumerate(dataset_item[answer]):
                                    dataset_item[answer][i] = answer_item.to(device)
                                    # print(answer_item)
                        elif ix == 8:  # to access the answers_lengths
                            dataset_item = dataset_item.to(device)
                        else:
                            for i, tensor_element in enumerate(dataset_item):
                                dataset_item[i] = tensor_element.to(device)
                                # print(tensor_element)

                    if idx % 10 == 0:
                        print('Test iter %d/%d' % (idx, len(test_generator)))
                    idx += 1

                    iter += 1
                    batch_n += 1
                    questions_encoded, labels, answers_encoded, vgg, c3d, questions_length, video_lengths, question_words, answers_lengths, qtexts, atexts = current_batch
                    answers_lengths = answers_lengths.to(device)

                    data_dict = {}
                    data_dict['video_features'] = [vgg, c3d]
                    data_dict['video_lengths'] = video_lengths
                    data_dict['question_words'] = question_words
                    data_dict['answers'] = labels
                    data_dict['question_lengths'] = questions_length
                    data_dict['num_mult_choices'] = 5
                    data_dict['answers_lengths'] = answers_lengths
                    data_dict['qtexts'] = qtexts
                    data_dict['atexts'] = atexts

                    model_output = rnn(data_dict)
                    outputs = model_output["outputs"]
                    predictions = model_output["predictions"]
                    targets = data_dict['answers']
                    len_data = len(vgg)

                    acc = rnn.accuracy(predictions, torch.stack(targets))
                    accuracy.update(acc.item(), len_data)

                    #writer.add_scalar('Test/accuracy (per example)', acc, idx)


                test_acc = accuracy.avg
                print('Test iter %d, acc %.3f' % (iter, accuracy.avg))

                writer.add_scalar('Test/accuracy (avg)', test_acc, iter)

                if best_test_acc < accuracy.avg:
                    best_test_acc = accuracy.avg
                    best_test_iter = iter

                print('[Test] iter %d, acc %.3f, best acc %.3f at iter %d' % (
                iter, test_acc, best_test_acc, best_test_iter))

                torch.save(rnn.state_dict(), os.path.join('.', args.save_model_path, 's%d' % (args.split,),
                                                          'rnn-%04d-vl_%.3f-t_%.3f.pkl' % (iter, val_acc, test_acc)))
                rnn.train()

    writer.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')

    path_saved_models = os.path.join('.', 'saved_models')
    parser.add_argument('--save_path', type=str, default=path_saved_models,
                        help='path for saving trained models')
    parser.add_argument('--split', type=int, help='which of the three splits to train/val/test, option: 0 | 1 | 2')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    parser.add_argument('--memory_type', type=str, help='_mrm2s | _stvqa | _enc_dec | _co_mem')
    parser.add_argument('--pretrain', type=str, help='y | n', default='n')
    parser.add_argument('--activation', type=int, help='0: Sigmoid |1:  Tanh |2: ReLU', default=1)
    parser.add_argument('--batch_size', type=int, help='a integer number', default=8)
    parser.add_argument('--learning_rate', type=float, help='a floating number', default=0.001)
    parser.add_argument('--ft_lr', type=float, help='a floating number', default=None)
    parser.add_argument('--hidden_size', type=int, help='an integer number', default=256)
    parser.add_argument('--epoch', type=int, help='an integer number', default=20)
    parser.add_argument('--embed_tech', type=str, help='glove | bert | elmo | xlm', default="glove")
    parser.add_argument('--aug_tech', type=int, help='1 to 6', default=-1)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--use_tp_att', type=bool, default=False)
    parser.add_argument('--use_int_mem', type=bool, default=False)
    parser.add_argument('--additional_tasks', type=str, default="")
    parser.add_argument('--dataset', type=str, default="egovqa", choices=["egovqa", "pororo"])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--QT_bias', action="store_true")
    parser.add_argument('--QT_weight', action="store_true")
    parser.add_argument('--skip_to', type=int, default=-1)

    args = parser.parse_args()

    it_gen = range(args.n_seeds) if args.skip_to < 0 else range(args.skip_to, args.n_seeds)

    if args.dataset == "egovqa":
        if args.split is None:
            for _s in [0, 1, 2]:
                args.split = _s
                for _seed in it_gen:
                    main(args, _seed)
        else:
            for _seed in it_gen:
                main(args, _seed)
    else:
        args.split = 0
        for _seed in it_gen:
            main(args, _seed)
