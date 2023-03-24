import os
import torch
import pickle as pkl
import numpy as np

_qtypes_map = {"action": 0,
               "person": 1,
               "abstract": 2,
               "detail": 3,
               "method": 4,
               "reason": 5,
               "location": 6,
               "statement": 7,
               "causality": 8,
               "yes_no": 9,
               "time": 10}

_inv_qtypes_map = {0: "action",
                   1: "person",
                   2: "abstract",
                   3: "detail",
                   4: "method",
                   5: "reason",
                   6: "location",
                   7: "statement",
                   8: "causality",
                   9: "yes_no",
                   10: "time"}

def get_type(q, a1, a2, a3, a4, a5):
    tp = None
    _q = q.lower()
    if _q.startswith("who") or "who " in _q:
        tp = "person"
    elif _q.startswith("is") \
        or False not in list(map(lambda t: "yes" in t or "no" in t, [a1, a2, a3, a4, a5])) \
        or _q.startswith("did") or _q.startswith("does") or _q.startswith("do ") or _q.startswith("was") \
        or _q.startswith("were") or _q.startswith("are"):
        tp = "yes_no"
    elif _q.startswith("where") or " where " in _q:
        tp = "location"
    elif _q.startswith("why"):
        tp = "reason"
    elif _q.startswith("when"):
        tp = "time"
    elif _q.startswith("how") or "how" in _q:
        tp = "method"
    elif _q.startswith("what"):
        if "happen" in _q:
            tp = "causality"
        elif "wear" in _q or "color" in _q or "colour" in _q or "body" in _q:
            tp = "detail"
        elif "saying" in _q or "tell" in _q:
            tp = "action"
        elif ("did" in _q and "say" in _q) or "said" in _q:
            tp = "statement"
        elif "think" in _q:
            tp = "abstract"
        elif "ask" in _q or "reply" in _q or "replied" in _q or "compliment" in _q:
            tp = "detail"
        elif "looking" in _q or "weather" in _q or "name" in _q or "see" in _q or "magic" in _q \
                or "attribute" in _q or "adjective" in _q or "song" in _q or "reason" in _q:
            tp = "abstract"
        elif "shape" in _q or "invention" in _q or "kind" in _q or "creature" in _q or "what is that" in _q \
                or "smell" in _q:
            tp = "detail"
        elif "title" in _q or "problem" in _q or "reaction" in _q:
            tp = "abstract"
        elif "making" in _q or "make" in _q or "trying to" in _q:
            tp = "action"
        elif "condition" in _q or "worried" in _q or "happy" in _q or "afternoon" in _q:
            tp = "abstract"
        elif "episode" in _q:
            tp = "detail"
        elif "surprise" in _q or "imagin" in _q:
            tp = "abstract"
        elif "doing" in _q or "play" in _q:
            tp = "action"
        elif "time" in _q:
            tp = "time"
        elif "decide" in _q or "command" in _q or ("did" in _q and "do" in _q) or ("does" in _q and "do" in _q) \
            or ("do" in _q and "do" in _q[_q.index("do"):]) or "going to" in _q or "did" in _q:
            tp = "action"
        elif "friend" in _q:
            tp = "person"
        else:
            tp = "abstract"
    elif "whom" in _q:
        tp = "person"
    elif "time" in _q or "night" in _q or "meanwhile" in _q:
        tp = "time"
    elif "place" in _q:
        tp = "location"
    elif "moral" in _q or "lesson" in _q:
        tp = "abstract"
    elif "friend" in _q:
        if True in (map(lambda x: "yes" in x, [a1, a2, a3, a4, a5])):
            tp = "yes_no"
        elif "feeling" in _q:
            tp = "abstract"
        elif "climb" in _q or "playing" in _q:
            tp = "detail"
        elif "way" in _q:
            tp = "location"
        else:
            tp = "person"
    elif "food" in _q:
        tp = "detail"
    elif "character" in _q:
        tp = "person"
    elif "happen" in _q:
        tp = "causality"
    elif "egg" in _q or "body" in _q or ("which" in _q and "team" in _q) or "instrument" in _q:
        tp = "detail"
    else:
        tp = "abstract"

    return tp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, path_feats,
                 aug_tech=-1, app_ft_vcs=True, mot_ft_vcs=True):
        assert app_ft_vcs and mot_ft_vcs, \
            "appearance/motion feature vectors: not implemented yet"

        self.video_names = labels
        self.list_IDs = list_IDs
        print(len(list_IDs))
        if aug_tech >= 0:
            augs_to_apply = _augs[aug_tech]
            for _f in augs_to_apply:
                self.list_IDs = _f(self.list_IDs, labels)

        supp_nums = {}
        tmp_supp_nums = {}
        for row in list_IDs.iterrows():
            l = row[1]
            if l["video_name"] not in tmp_supp_nums.keys():
                tmp_supp_nums[l["video_name"]] = set()
            tmp_supp_nums[l["video_name"]].add(l["supporting_num"])
        for vk in tmp_supp_nums.keys():
            supp_nums[vk] = list(tmp_supp_nums[vk])

        self.vgg_dict = {}
        self.c3d_dict = {}

        with open("data_pororo/vocab_pororo.txt") as f:
            w2i = {}
            vocab = f.readlines()
            for _i, w in enumerate(vocab):
                w2i[w.strip()] = _i
            self.word2int = w2i

        for r in list_IDs.iterrows():
            vc = r[1]["video_name"]
            for sn in supp_nums[vc]:
                _vc = vc.replace(",", "")
                try:
                    with open(os.path.join(path_feats, "vgg_{}_{}.pkl".format(_vc, sn)), "rb") as f:
                        feat1 = pkl.load(f, encoding='latin1')
                        feat1 = feat1['feat']
                        if _vc not in self.vgg_dict.keys():
                            self.vgg_dict[_vc] = {}
                        self.vgg_dict[_vc][sn] = feat1
                except:
                    print("vgg_{}_{}.pkl not found".format(_vc, sn))
                    exit()

                try:
                    with open(os.path.join(path_feats, "c3d_{}_{}.pkl".format(_vc, sn)), "rb") as f:
                        feat1 = pkl.load(f, encoding='latin1')
                        feat1 = feat1['feat']
                        if _vc not in self.c3d_dict.keys():
                            self.c3d_dict[_vc] = {}
                        self.c3d_dict[_vc][sn] = feat1
                except:
                    print("c3d_{}_{}.pkl not found".format(_vc, sn))
                    exit()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs.loc[index]

        v_n = ID["video_name"].replace(",", "")
        vgg_feat = self.vgg_dict[v_n][ID["supporting_num"]]
        c3d_feat = self.c3d_dict[v_n][ID["supporting_num"]]
        vgg_feat = torch.from_numpy(vgg_feat).unsqueeze(0)
        c3d_feat = torch.from_numpy(c3d_feat).unsqueeze(0)

        question_length = len(ID['question'].split())

        import string
        q_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["question"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        q_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["question"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a1_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["answer0"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a1_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["answer0"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a2_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["answer1"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a2_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["answer1"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a3_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["answer2"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a3_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["answer2"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a4_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["answer3"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a4_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["answer3"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a5_enc = ",".join([str(self.word2int[w.translate(str.maketrans("", "", string.punctuation))])
                          for w in ID["answer4"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])
        a5_txt = " ".join([w.translate(str.maketrans("", "", string.punctuation))
                          for w in ID["answer4"].split()
                          if w.translate(str.maketrans("", "", string.punctuation)) != ""])

        _texts = {"qtext": q_txt,
                  "a1": a1_txt,
                  "a2": a2_txt,
                  "a3": a3_txt,
                  "a4": a4_txt,
                  "a5": a5_txt
                  }

        return q_enc,\
               ID['correct_idx'],\
               a1_enc,\
               a2_enc,\
               a3_enc,\
               a4_enc,\
               a5_enc,\
               vgg_feat,\
               c3d_feat,\
               question_length,\
               q_txt,\
               ID["answer{}".format(ID["correct_idx"])],\
               _texts

if __name__ == "__main__":
    import pandas
    import json
    df = json.load(open("data_pororo/pororo_qa.json"))
    s = [get_type(q["question"], q["answer0"], q["answer1"], q["answer2"], q["answer3"], q["answer4"]) for q in
         df["PororoQA"]]
    r = pandas.Series(s).value_counts()
    print("====")

    _a = {}
    for a in r.items():
        _a[a[0]] = a[1]

    for x in ["action", "person", "abstract", "detail", "method", "reason", "location", "statement", "causality",
              "yes_no", "time"]:
        c = 0 if x not in _a.keys() else _a[x]
        print("{} {} -> {:.2f}".format(x, c, c / len(s)))
    for x in r.keys():
        if x not in ["action", "person", "abstract", "detail", "method", "reason", "location", "statement", "causality",
                     "yes_no", "time"]:
            c = 0 if x not in _a.keys() else _a[x]
            print("{} {} -> {:.2f}".format(x, c, c / len(s)))

    q_col = 0
    for q in df["PororoQA"]:
        if "color" in q["question"] or "colour" in q["question"]:
            q_col += 1

    print("found {} questions containing 'colo[u]r'".format(q_col))


    def get_len(PIL_Image_object):
        """ Returns the length of a PIL Image object """
        PIL_Image_object.seek(0)
        frames = duration = 0
        while True:
            try:
                frames += 1
                duration += PIL_Image_object.info['duration']
                PIL_Image_object.seek(PIL_Image_object.tell() + 1)
            except EOFError:
                return duration
        return None

    clips = set()
    eps = set()
    clip_lens = []
    for l in df["PororoQA"]:
        eps.add(l["video_name"])
        clips.add("_".join([l["video_name"], l["supporting_num"]]))
        season = "_".join(l["video_name"].replace(",", "").split("_")[:-1])
        clip_fs = os.path.join("..",
                               "Scenes_Dialogues",
                               season,
                               l["video_name"].replace(",", ""),
                               "{}.gif".format(l["supporting_num"]))
        from PIL import Image

        clip_gif = Image.open(clip_fs)
        c_len = get_len(clip_gif)
        #print("{} -> {}".format("_".join([l["video_name"], l["supporting_num"]]), c_len))
        clip_lens.append(c_len)
    clip_lens = np.array(clip_lens)
    print("there are {} clips and {} episodes".format(len(clips), len(eps)))
    print("average length {} (ms)".format(clip_lens.mean()))

