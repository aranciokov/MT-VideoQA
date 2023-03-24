# Video question answering supported by a multi-task learning objective

In this repo, we provide code for the paper **Video question answering supported by a multi-task learning objective**, which was accepted at [Multimedia Tools and Applications](https://link.springer.com/article/10.1007/s11042-023-14333-0).

#### Python environment
Requirements: allennlp==2.0.0 (for elmo), torch==1.7.0, torchvision==0.8.1, tensorboard==2.4.0, pandas==1.1.5, numpy==1.18.5, tables, colorlog, protobuf==3.20.0, transformers==4.2.2 \*

_\*: The version 2 of allennlp is quite old now, so installing it now may lead to some dependency problems with the subsequent packages. In particular, it seems to create some problems with the original version of transformers we used (3.5.1): installing the 4.2.2 seems to work fine._

#### Data folders
- Pororo: [here](https://drive.google.com/file/d/1RuK_WWBreqwzbFwFDru93apJETCNgO58/view?usp=sharing)
- EgoVQA: [here](https://drive.google.com/file/d/1dwKC2iWGdXFSSCjgA90h7MQDegk_o95k/view?usp=sharing)

#### Training
To launch a training, run

``python train_dataset.py --memory_type M --embed_tech E --dataset D``

where:
- M in '_stvqa', '_co_mem', '_mrm2s'
- E in 'glove', 'bert', 'elmo', 'xlm'
- D in 'pororo', 'egovqa'

If you want to test the multi-task learning objective, add ``--additional_tasks qtclassif``

To gather the evaluation statistics across multiple runs, call ``test_dataset.py`` in place of the train one, and add ``--save_path saved_models``

#### Acknowledgements
We thank the authors of [Fan, (ICCVW, 2019)](https://openaccess.thecvf.com/content_ICCVW_2019/html/EPIC/Fan_EgoVQA_-_An_Egocentric_Video_Question_Answering_Benchmark_Dataset_ICCVW_2019_paper.html) ([github](https://github.com/fanchenyou/EgoVQA))
 for the release of their codebase. 

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@article{falcon2022multitask,
  title={Video question answering supported by a multi-task learning objective},
  author={Falcon, Alex and Serra, Giuseppe and Lanz, Oswald},
  journal={Multimedia Tools and Applications},
  year={2023},
  organization={Springer}
}
```

## License

MIT License
