# coding=utf-8
import os
import argparse
import json
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np

from dataset_pytorch import Dataset as EgoVQA_Dataset
from dataset_pororo import Dataset as Pororo_Dataset
from embed_loss import MultipleChoiceLoss
from attention_module import *
from util import AverageMeter
from collections import defaultdict

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

    if args.dataset == "egovqa":
        path_data_split = os.path.join('.', 'data', 'data_split.json')
        with open(path_data_split, 'r') as f:
            splits = json.load(f)
    else:
        video_names = list()

        for l in json.load(open("data_pororo/pororo_qa.json"))["PororoQA"]:
            if l["video_name"] not in video_names:
                video_names.append(l["video_name"])
        video_names = sorted(video_names,
                             key=lambda n: int(n.split("_ep")[1].replace(",", "")))  # list(sorted(video_names))

    args.image_feature_net = 'concat'
    args.layer = 'fc'

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
                     use_int_mem=args.use_int_mem, device=device, dataset=args.dataset, return_features=True,
                     **kwargs)
    rnn = rnn.to(device)

    print(rnn)

    count_type = defaultdict(int)
    count_correct = defaultdict(int)
    accuracy = AverageMeter()
    correct = 0
    correct_qt_pred = 0
    total_qs = 0

    split_acc = []
    split_acc2 = []

    use_pretrain = True if args.pretrain == 'y' else False

    if args.dataset == "egovqa":
        path_data = os.path.join('.', 'data')
        path_data_feats = os.path.join(path_data, 'feats')
    else:
        path_data = 'data_pororo'
        path_data_feats = os.path.join(path_data, 'feats_pororo')

    split_ns = [0, 1, 2] if args.dataset == "egovqa" else [0]

    from torch.utils.tensorboard import SummaryWriter
    tsne_log_path = 'tsne_{}{}_{}{}_{}_bs{}_lr{}{}_seed{}{}{}{}{}{}'.format(
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
    )
    if args.save_embeddings or args.single_point_embeds:
        logger = SummaryWriter(log_dir=f"test_embeddings/{tsne_log_path}")

    for sp in split_ns:
        if args.dataset == "egovqa":
            video_cam_split = splits[sp]
        else:
            video_cam_split = [video_names[:94], video_names[94:137],
                               video_names[137:]]


        batch_size = args.batch_size
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
            print("loaded pororoQA", len(total_qa))

        # partition =

        partition = {'train': [], 'validation': [], 'test': []}
        # labels = []

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
            print(len(partition["train"]), len(partition["validation"]), len(partition["test"]))

        if args.dataset == "egovqa":
            dts = EgoVQA_Dataset
        else:
            dts = Pororo_Dataset

        use_precomp_fts = True
        if args.memory_type in ["_mrm2s_convLSTM", "_mrm2s_FTFCs"]:
            use_precomp_fts = False

        test_set = dts(partition['test'], video_cam_split[2], path_data_feats,
                           app_ft_vcs=use_precomp_fts, mot_ft_vcs=use_precomp_fts)
        test_generator = torch.utils.data.DataLoader(test_set, **params)

        if not use_pretrain:
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
            # args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net, args.layer, args.memory_type)
        else:
            assert False, "args.pretrain == y, need to implement"
            args.save_model_path = os.path.join(args.save_path,
                                                'model_%s_%s_withpretrain_%s_%s_bs%s_lr%s_seed%d' % (
                                                args.image_feature_net, args.layer,
                                                args.memory_type, args.embed_tech,
                                                (str(args.batch_size)), str(args.learning_rate), seed)
                                                )
            # args.save_model_path = args.save_path + 'model_%s_%s_withpretrain_%s' % (
            # args.image_feature_net, args.layer, args.memory_type)

        output_features_for_tsne = []
        labels_for_tsne = []

        args.save_model_path = os.path.join(args.save_model_path, '%s_hidden_size%s' % (
            name_activation_function, str(args.hidden_size)))

        args.save_model_path = os.path.join(args.save_model_path, 's%d' % (sp))
        resume_file_path = {0: args.resume_file_s0, 1: args.resume_file_s1, 2: args.resume_file_s2}[sp]
        print("looking for models in {}".format(args.save_model_path))
        if resume_file_path == "":
            files = sorted(os.listdir(args.save_model_path))  # protocol 1: only first 20 models
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
            split_acc.append(max_test)
            rnn.load_state_dict(torch.load(os.path.join(args.save_model_path, best_state), map_location=args.device))

        elif args.memory_type=="_mrm2s":
            print("resuming file path: {}".format(resume_file_path))
            max_val = float(resume_file_path.split("/")[-1].split("-")[2].split("_")[1])
            max_test = float(resume_file_path.split("/")[-1].split("-")[3].split("_")[1].replace(".pkl", ""))
            print("load checkpoint: {}, {}".format(max_val, max_test))
            split_acc.append(max_test)
            weights = torch.load(resume_file_path)
            _ = weights.pop("video_encoder.weight")
            _ = weights.pop("video_encoder.bias")
            weights["linear_mem.weight"] = weights.pop("linear_decoder_mem.weight")
            weights["linear_mem.bias"] = weights.pop("linear_decoder_mem.bias")
            weights["hidden_enc_1.weight"] = weights.pop("hidden_encoder_1.weight")
            weights["hidden_enc_1.bias"] = weights.pop("hidden_encoder_1.bias")
            weights["hidden_enc_2.weight"] = weights.pop("hidden_encoder_2.weight")
            weights["hidden_enc_2.bias"] = weights.pop("hidden_encoder_2.bias")
            weights["linear_att_a.weight"] = weights.pop("linear_decoder_att_a.weight")
            weights["linear_att_a.bias"] = weights.pop("linear_decoder_att_a.bias")
            weights["linear_att_m.weight"] = weights.pop("linear_decoder_att_m.weight")
            weights["linear_att_m.bias"] = weights.pop("linear_decoder_att_m.bias")
            weights["linear2.weight"] = weights.pop("linear_decoder_count_2.weight")
            weights["linear2.bias"] = weights.pop("linear_decoder_count_2.bias")

            weights["lstm_mm_1.weight_ih_l0"] = weights.pop("lstm_mm_1.weight_ih")
            weights["lstm_mm_1.bias_ih_l0"] = weights.pop("lstm_mm_1.bias_ih")
            weights["lstm_mm_1.weight_hh_l0"] = weights.pop("lstm_mm_1.weight_hh")
            weights["lstm_mm_1.bias_hh_l0"] = weights.pop("lstm_mm_1.bias_hh")
            weights["lstm_mm_2.weight_ih_l0"] = weights.pop("lstm_mm_2.weight_ih")
            weights["lstm_mm_2.bias_ih_l0"] = weights.pop("lstm_mm_2.bias_ih")
            weights["lstm_mm_2.weight_hh_l0"] = weights.pop("lstm_mm_2.weight_hh")
            weights["lstm_mm_2.bias_hh_l0"] = weights.pop("lstm_mm_2.bias_hh")

            weights["lstm_text_1.weight_ih_l0"] = weights.pop("lstm_text_1.weight_ih")
            weights["lstm_text_1.bias_ih_l0"] = weights.pop("lstm_text_1.bias_ih")
            weights["lstm_text_1.weight_hh_l0"] = weights.pop("lstm_text_1.weight_hh")
            weights["lstm_text_1.bias_hh_l0"] = weights.pop("lstm_text_1.bias_hh")
            weights["lstm_text_2.weight_ih_l0"] = weights.pop("lstm_text_2.weight_ih")
            weights["lstm_text_2.bias_ih_l0"] = weights.pop("lstm_text_2.bias_ih")
            weights["lstm_text_2.weight_hh_l0"] = weights.pop("lstm_text_2.weight_hh")
            weights["lstm_text_2.bias_hh_l0"] = weights.pop("lstm_text_2.bias_hh")
            weights["lstm_video_1_a.weight_ih_l0"] = weights.pop("lstm_video_1a.weight_ih")
            weights["lstm_video_1_a.bias_ih_l0"] = weights.pop("lstm_video_1a.bias_ih")
            weights["lstm_video_1_a.weight_hh_l0"] = weights.pop("lstm_video_1a.weight_hh")
            weights["lstm_video_1_a.bias_hh_l0"] = weights.pop("lstm_video_1a.bias_hh")
            weights["lstm_video_2_a.weight_ih_l0"] = weights.pop("lstm_video_2a.weight_ih")
            weights["lstm_video_2_a.bias_ih_l0"] = weights.pop("lstm_video_2a.bias_ih")
            weights["lstm_video_2_a.weight_hh_l0"] = weights.pop("lstm_video_2a.weight_hh")
            weights["lstm_video_2_a.bias_hh_l0"] = weights.pop("lstm_video_2a.bias_hh")
            weights["lstm_video_1_m.weight_ih_l0"] = weights.pop("lstm_video_1m.weight_ih")
            weights["lstm_video_1_m.bias_ih_l0"] = weights.pop("lstm_video_1m.bias_ih")
            weights["lstm_video_1_m.weight_hh_l0"] = weights.pop("lstm_video_1m.weight_hh")
            weights["lstm_video_1_m.bias_hh_l0"] = weights.pop("lstm_video_1m.bias_hh")
            weights["lstm_video_2_m.weight_ih_l0"] = weights.pop("lstm_video_2m.weight_ih")
            weights["lstm_video_2_m.bias_ih_l0"] = weights.pop("lstm_video_2m.bias_ih")
            weights["lstm_video_2_m.weight_hh_l0"] = weights.pop("lstm_video_2m.weight_hh")
            weights["lstm_video_2_m.bias_hh_l0"] = weights.pop("lstm_video_2m.bias_hh")
            rnn.load_state_dict(weights)
        else:
            print("resuming file path: {}".format(resume_file_path))
            max_val = float(resume_file_path.split("/")[-1].split("-")[2].split("_")[1])
            max_test = float(resume_file_path.split("/")[-1].split("-")[3].split("_")[1].replace(".pkl", ""))
            print("load checkpoint: {}, {}".format(max_val, max_test))
            split_acc.append(max_test)
            weights = torch.load(resume_file_path)
            rnn.load_state_dict(weights)

        with torch.no_grad():
            rnn.eval()
            correct_sp = 0
            idx = 0
            model_answers = []
            questions = []
            target_answers = []
            qt_predicted = []
            for current_batch in test_generator:
                for ix, dataset_item in enumerate(current_batch):
                    if ix == 2:  # to access the dictionary containing the possible answers
                        for answer in dataset_item:
                            for i, answer_item in enumerate(dataset_item[answer]):
                                dataset_item[answer][i] = answer_item.to(device)
                                # print(answer_item)
                    elif ix == 8:  # to access the answers_lengths
                        dataset_item = dataset_item.to(device)
                    elif ix == 9 or ix == 10:
                        pass
                    else:
                        for i, tensor_element in enumerate(dataset_item):
                            dataset_item[i] = tensor_element.to(device)
                            # print(tensor_element)

                if idx % 100 == 0:
                    print('Test iter %d/%d' % (idx, len(test_generator)))
                idx += 1

                questions_encoded, labels, answers_encoded, vgg, c3d, questions_length, video_lengths, question_words, answers_lengths, qtext, atext = current_batch
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
                data_dict['qtexts'] = qtext
                data_dict['atexts'] = atext
                model_output = rnn(data_dict)
                outputs = model_output["outputs"]
                predictions = model_output["predictions"]
                out_features = model_output["features"]
                #print(outputs, predictions)
                targets = torch.stack(data_dict['answers'])

                output_features_for_tsne.append(out_features)
                labels_for_tsne += [f"{qtext[0]}? {a[0]}" for a in atext.values()]

                #print(predictions, targets)
                acc = rnn.accuracy(predictions, targets)
                accuracy.update(acc.item(), len_data)

                prediction = predictions.item()
                target = targets.item()

                if args.dataset == "egovqa":
                    tp = None
                    assert len(qtext) == 1, "qtext with more than 1 elem: {}".format(qtext)
                    qtext = qtext[0]
                    qtext = qtext.lower()
                    if 'am i' in qtext and 'doing' in qtext:
                        # first-person action
                        tp = '1_action_1st'
                    elif 'doing' in qtext:
                        # third-person action
                        tp = '2_action_3rd'
                    elif 'how many' in qtext:
                        tp = '5_count'
                    elif 'color' in qtext and 'what' in qtext:
                        tp = '6_what_color'
                    elif 'what' in qtext:
                        if 'am i' in qtext or 'my' in qtext:
                            tp = '3_what_obj_1st'
                        else:
                            tp = '3_what_obj_3rd'
                    elif qtext.startswith('who'):
                        if 'me' in qtext or 'am i' in qtext:
                            tp = '4_who_1st'
                        else:
                            tp = '4_who_3rd'
                    else:
                        tp = '7_other'

                else:
                    from dataset_pororo import get_type
                    tp = get_type(qtext[0], atext["a1"][0], atext["a2"][0], atext["a3"][0], atext["a4"][0], atext["a5"][0])
                # print tp, ' ', qtext, ' ', atext, prediction

                count_type[tp] += 1
                if prediction == target:
                    correct += 1
                    correct_sp += 1
                    count_correct[tp] += 1

                # ---- additional tasks ----
                if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(","):
                    qt_logits = model_output["qt_classes"]
                    if args.dataset == "egovqa":
                        from dataset_pytorch import _inv_qtypes_map
                    else:
                        from dataset_pororo import _inv_qtypes_map
                    _, pred_qt = torch.max(qt_logits, -1)
                    pred_qt = list(map(lambda q: _inv_qtypes_map[q], pred_qt.squeeze(0).cpu().numpy()))
                    _cor_pred_qts = list(filter(lambda e: e, list(map(lambda t: t == tp, pred_qt))))
                    correct_qt_pred += len(_cor_pred_qts)
                    total_qs += len(pred_qt)
                # ---- additional tasks ----

                if tp == "6_what_color":
                    if args.single_point_embeds:
                        print(["{}{} -> {:.3f}".format((('(+)' if outputs[0].argmax().item() == labels[0].item() else '(-)') if i == labels[0].item() else ''),
                                                   atext[a][0],
                                                   v.item()) for a, v, i in zip(atext, outputs[0], torch.arange(len(atext)))])

                model_answers.append(atext[f"a{prediction+1}"])
                questions.append(qtext)
                target_answers.append(atext[f"a{target+1}"])
                if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(","):
                    qt_predicted.append(pred_qt)

            test_acc = 1.0 * correct_sp / len(test_generator)
            print(correct, len(test_generator))
            print('Test acc %.3f' % (test_acc,))
            split_acc2.append(test_acc)

        if args.save_embeddings:
            logger.add_embedding(torch.cat(output_features_for_tsne), metadata=labels_for_tsne, tag=f"split{sp}")

        name_file = f"results_testing_qualitative_{args.dataset}_s{sp}.csv"
        if not os.path.exists(name_file):
            with open(name_file, "w") as f:
                f.write("question,groundtruth\n")
                for a, q in zip(target_answers, questions):
                    f.write(f"{q},{a}\n")

        with open(name_file, "r") as f:
            header = f.readline()
            lines = f.readlines()
        with open(name_file, "w") as f:
            f.write(
                f"{header.strip()},{'{}_{}{}'.format(args.memory_type, args.embed_tech, '+QTC' if 'qtclassif' in args.additional_tasks else '')}\n")
            for j, (l, a) in enumerate(zip(lines, model_answers)):
                assert not l.startswith("question")
                if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(","):
                    f.write(f"{l.strip()},{f'{a} (QT:{qt_predicted[j]})'}\n")
                else:
                    f.write(f"{l.strip()},{a}\n")

    if args.save_embeddings:
        logger.close()

    class_acc = 0.0
    question_accuracy = []
    print('Question types accuracy: '),
    for tp in sorted(count_type.keys()):
        a = count_correct[tp]
        b = count_type[tp]
        if b == 0:
            b = 1
        print('(', tp, a, b, 1.0 * a / b, ') ')
        class_acc += 1.0 * a / b
    print()
    print('Question types accuracy: '),
    for tp in sorted(count_type.keys()):
        a = count_correct[tp]
        b = count_type[tp]
        if b == 0:
            b = 1
        question_accuracy.append(round(100.0 * a / b, 2))
        print(round(100.0 * a / b, 2), '& '),

    print()
    print(count_type)
    print('Question types per class accuracy: ', class_acc / len(count_type.keys()))

    print('split acc', split_acc, split_acc2)
    if args.dataset == "egovqa":
        print('%.2f & %.2f & %.2f & %.2f' % (
        split_acc[0], split_acc[1], split_acc[2], (split_acc[0] + split_acc[1] + split_acc[2]) / 3,))

        questions_accuracy = str(question_accuracy)[1:len(str(question_accuracy)) - 1]
        f = open("{}results_testing.csv".format("pororo_" if args.dataset == "pororo" else ""), "a")
        f.write(str(seed) + "," +
                str(args.aug_tech) + "," +
                args.additional_tasks + "," +
                ("Y" if args.use_int_mem else "") + "," +
                args.memory_type + "," +
                args.embed_tech + "," +
                name_activation_function + "," +
                str(batch_size) + "," +
                str(args.hidden_size) + "," +
                str(args.learning_rate) + "," +
                ("," if "_ft" not in args.embed_tech else "{},".format(args.ft_lr)) +
                ("{:.2f}".format(correct_qt_pred / total_qs)
                 if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",")
                 else "") + "," +
                questions_accuracy + "," +
                ('%.2f, %.2f, %.2f, %.2f' % (
                    split_acc[0], split_acc[1], split_acc[2], (split_acc[0] + split_acc[1] + split_acc[2]) / 3,)) +
                " \n")
        f.close()

        f = open("{}results_table.txt".format("pororo_" if args.dataset == "pororo" else ""), "a")
        f.write(str(seed) + " & " +
                str(args.aug_tech) + " & " +
                args.additional_tasks + " & " +
                ("Y" if args.use_int_mem else "") + " & " +
                args.memory_type + " & " +
                args.embed_tech + " & " +
                name_activation_function + " & " +
                str(batch_size) + " & " +
                str(args.hidden_size) + " & " +
                str(args.learning_rate) + " & " +
                (" & " if "_ft" not in args.embed_tech else "{} & ".format(args.ft_lr)) +
                ("{:.2f}".format(correct_qt_pred / total_qs)
                 if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",")
                 else "") + " & " +
                questions_accuracy + " & " +
                ('%.2f & %.2f & %.2f & %.2f' % (
                    split_acc[0], split_acc[1], split_acc[2], (split_acc[0] + split_acc[1] + split_acc[2]) / 3,)) +
                " \\\\ \n")
        f.close()
    else:
        print("acc {}".format(split_acc[0]))
        questions_accuracy = str(question_accuracy)[1:len(str(question_accuracy)) - 1]
        f = open("{}results_testing.csv".format("pororo_" if args.dataset == "pororo" else ""), "a")
        f.write(str(seed) + "," +
                str(args.aug_tech) + "," +
                args.additional_tasks + "," +
                ("Y" if args.use_int_mem else "") + "," +
                args.memory_type + "," +
                args.embed_tech + "," +
                name_activation_function + "," +
                str(batch_size) + "," +
                str(args.hidden_size) + "," +
                str(args.learning_rate) + "," +
                ("," if "_ft" not in args.embed_tech else "{},".format(args.ft_lr)) +
                ("{:.2f}".format(correct_qt_pred / total_qs)
                 if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",")
                 else "") + "," +
                questions_accuracy + "," +
                ('%.2f' % (split_acc[0])) +
                " \n")
        f.close()

        f = open("{}results_table.txt".format("pororo_" if args.dataset == "pororo" else ""), "a")
        f.write(str(seed) + " & " +
                str(args.aug_tech) + " & " +
                args.additional_tasks + " & " +
                ("Y" if args.use_int_mem else "") + " & " +
                args.memory_type + " & " +
                args.embed_tech + " & " +
                name_activation_function + " & " +
                str(batch_size) + " & " +
                str(args.hidden_size) + " & " +
                str(args.learning_rate) + " & " +
                (" & " if "_ft" not in args.embed_tech else "{} & ".format(args.ft_lr)) +
                ("{:.2f}".format(correct_qt_pred / total_qs)
                 if args.additional_tasks != "" and "qtclassif" in args.additional_tasks.split(",")
                 else "") + " & " +
                questions_accuracy + " & " +
                ('%.2f' % (split_acc[0])) +
                " \\\\ \n")
        f.close()

    return question_accuracy, split_acc

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    path_saved_models = os.path.join('.', 'saved_models')
    parser.add_argument('--save_path', type=str, default=path_saved_models,
                        help='path for saving trained models')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    parser.add_argument('--memory_type', type=str, help='_mrm2s | _stvqa | _enc_dec | _co_mem')
    parser.add_argument('--pretrain', type=str, help='y | n', default='n')
    parser.add_argument('--activation', type=int, help='0: Sigmoid |1:  Tanh |2: ReLU', default=1)
    parser.add_argument('--batch_size', type=int, help='a integer number', default=8)
    parser.add_argument('--learning_rate', type=float, help='a floating number', default=0.001)
    parser.add_argument('--ft_lr', type=float, help='a floating number', default=0.001)
    parser.add_argument('--hidden_size', type=int, help='an integer number', default=256)
    parser.add_argument('--epoch', type=int, help='an integer number', default=20)
    parser.add_argument('--embed_tech', type=str, help='glove | bert | elmo | xlm', default="glove")
    parser.add_argument('--aug_tech', type=int, help='1 to 6', default=-1)
    parser.add_argument('--use_int_mem', type=bool, default=False)
    parser.add_argument('--additional_tasks', type=str, default="")
    parser.add_argument('--resume_file_s0', type=str, default="")  
    parser.add_argument('--resume_file_s1', type=str, default="")  
    parser.add_argument('--resume_file_s2', type=str, default="")  
    parser.add_argument('--dataset', type=str, default="egovqa")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--QT_bias', action="store_true")
    parser.add_argument('--QT_weight', action="store_true")
    parser.add_argument('--save_embeddings', action="store_true")
    parser.add_argument('--single_point_embeds', action="store_true")
    parser.add_argument('--n_seeds', type=int, default=5)

    args = parser.parse_args()
    n_seeds = args.n_seeds

    qt_accs, spl_accs = [], []
    for _seed in range(n_seeds):
        qt_acc, spl_acc = main(args, _seed)
        qt_accs.append(qt_acc)
        spl_accs.append(spl_acc)

    qt_accs = np.array(qt_accs)
    if args.dataset == "egovqa":
        qt_accs_avg = qt_accs.mean(0)[:-1]
        qt_accs_std = qt_accs.std(0)[:-1]
    else:
        qt_accs_avg = qt_accs.mean(0)
        qt_accs_std = qt_accs.std(0)
    qt_str = " & ".join(["${:.2f}_{{\\pm {:.2f}}}$".format(_avg, _std) for _avg, _std in zip(qt_accs_avg, qt_accs_std)])

    spl_accs = np.array(spl_accs)
    spl_accs_avg = spl_accs.mean(0)
    spl_accs_std = spl_accs.std(0)
    spl_str = " & ".join(["${:.2f}_{{\\pm {:.2f}}}$".format(_avg, _std) for _avg, _std in zip(spl_accs_avg, spl_accs_std)])
    name_activation_function = str(args.activation_function).split('.')[-1].split('\'')[0]
    f = open("{}results_table_avgs.txt".format("pororo_" if args.dataset == "pororo" else ""), "a")
    f.write(str(n_seeds) + " & " +
            str(args.aug_tech) + " & " +
            args.additional_tasks + " & " +
            ("Y" if args.use_int_mem else "") + " & " +
            args.memory_type + " & " +
            args.embed_tech + " & " +
            name_activation_function + " & " +
            str(args.batch_size) + " & " +
            str(args.hidden_size) + " & " +
            str(args.learning_rate) + " & " +
            (" & " if "_ft" not in args.embed_tech else "{} & ".format(args.ft_lr)) +
            qt_str + " & " +
            spl_str + " & " +
            "${:.2f}_{{\\pm {:.2f}}}$".format(spl_accs.mean(), spl_accs.std()) + " " +
            " \\\\ \n")
    f.close()

    arch_names = {"_stvqa": "ST-VQA", "_enc_dec": "ST-VQA w/o Att", "_mrm2s": "HME-VQA", "_co_mem": "Co-Mem"}
    embed_names = {"bert": "BERT", "elmo": "ELMo", "glove_frozen": "GloVe", "xlm": "XLM",
                   "bert_ft": "BERT*", "elmo_ft": "ELMo*", "xlm_ft": "XLM*", "glove": "GloVe*", "glove_ft": "GloVe*",
                   "bert_jointft": "BERT* (jointly trained)"}

    f = open("{}results_table1_avgs.txt".format("pororo_" if args.dataset == "pororo" else ""), "a")
    f.write(str(args.aug_tech) + " & " +
            args.additional_tasks + " & " +
            arch_names[args.memory_type] + " & " +
            embed_names[args.embed_tech] + " & " +
            (" & " if "_ft" not in args.embed_tech else "{} & ".format(args.ft_lr)) +
            spl_str + " & " +
            "${:.2f}_{{\\pm {:.2f}}}$".format(spl_accs.mean(), spl_accs.std()) + " " +
            " \\\\ \n")
    f.close()

    f = open("{}results_table2_avgs.txt".format("pororo_" if args.dataset == "pororo" else ""), "a")
    f.write(str(args.aug_tech) + " & " +
            args.additional_tasks + " & " +
            arch_names[args.memory_type] + " & " +
            embed_names[args.embed_tech] + " & " +
            (" & " if "_ft" not in args.embed_tech else "{} & ".format(args.ft_lr)) +
            qt_str + " " +
            " \\\\ \n")
    f.close()
