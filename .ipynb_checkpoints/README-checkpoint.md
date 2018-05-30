# korean-text-classification-tf
author : 박성남


## Objective
한국어에 다양한 language model들을 tensorflow를 활용하여 실험해보기 위해 만들었습니다. 그 중 text classification 문제입니다.

## Structure
- `data_helper` : 데이터를 불러오고, batch size에 따라 나눠주고, 정해진 길이의 sequence로 padding하고 전체 전처리를 관리하는 **Preprocessor**가 포함되어 있습니다.
- `normalizers` : 아주 기본적인 한국어 normalizer입니다. 욕의 변형 형태를 잡아내어 원래 욕으로 바꾸고, 필요없는 문장 기호를 없앱니다.
- `tokenizers` : 한국어 문장을 token으로 나눠줍니다.
    - `JamoTokenizer` : 문장을 자모 단위로 나눠줍니다.
    - `TwitterTokenizer` : KoNLP의 Twitter 패키지를 활용하여 문장을 tokenize합니다.
    - `SoyNLPTokenizer` : [SoyNLP](https://github.com/lovit/soynlp)의 MaxScoreTokenizer를 활용하여 문장을 tokenize합니다.
- `vectorizers` : token을 계산할 수 있는 vector로 바꿔줍니다.

## Models
- `TextCNN` : Convolutional Neural Networks for Sentence Classification [paper](https://arxiv.org/abs/1408.5882) [github](https://github.com/dennybritz/cnn-text-classification-tf)
- `VDCNN` : Very Deep Convolutional Networks for Text Classification [paper](https://arxiv.org/abs/1606.01781) [github](https://github.com/zonetrooper32/VDCNN)

## How to use
### Task, Data loading 관련 parameter
|Parameter      |Description                   |Default|
| ------------- |:-----------------------------:|:-------:|
| mode      | 지금은 train 밖에 없음 | train |
| small|잘 작동하는지 test용입니다. True이면 500개에 대해서만 돌아감|False|
| train_dir|모델을 학습할 train_set의 위치|./data/train.txt|
| val_dir|모델의 성능을 평가할 val_set의 위치|./data/test.txt|
| pretrained_embed_dir|pretrained된 embedding을 쓸 경우 그 위치. 없으면 random값으로 초기화|""|
| checkpoint_dir|모델을 저장하고 tensorboard로 확인할 폴더를 생성할 위치|""|

### Model 관련 parameter
|Parameter      |Description                   |Default|
| ------------- |:-----------------------------:|:-------:|
| model|사용할 모델 이름|TextCNN|
| normalizer|한국어의 오타를 수정하거나 유사한 단어를 하나의 단어로 바꿔줌|BasicNormalizer|
| tokenizer|한국어 문장을 token으로 나눠줌. 반드시 pre-trained embedding과 같이 생각해야함|JamoTokenizer|
| vocab_size|모든 단어를 다 쓸 수 없기 때문에, 이 크기까지만 unique token을 뽑고 나머지는 다 [UNK] 처리함|20000|
| embed_dim|embedding vector의 차원, 만약 pre-trained embedding을 쓴다면 그 차원에 맞춰야함|128|
| min_length|문장의 최소 토큰의 갯수, 그 이하의 문장은 삭제|64|
| max_length|문장의 최대 토큰의 갯수, 그 이상의 문장은 자름|128|
| dropout_keep_prob|regularization을 위한 dropout rate|0.5|

### Training 관련 parameter
|Parameter      |Description                   |Default|
| ------------- |:-----------------------------:|:-------:|
| batch_size|1 epoch당 batch size의 크기|64|
| num_epochs|최대 epoch의 크기|200|
| evaluate_every|몇 번마다 val_set으로 성능 평가할 것인지|1|
| checkpoint_every|몇 번마다 모델을 저장할 것인지|1|
| num_checkpoints|최근 몇 개의 모델까지 저장할 것인지|30|
| shuffle|batch로 나누기 전에 train_set을 shuffle할 것인지|False|

### Example
~~~
python3 train.py --mode=train --model=TextCNN --normalizer=BasicNormalizer --tokenizer=JamoTokenizer --vocab_size=20000 --embed_dim=128 --min_length=64 --max_length=512 --dropout_keep_prob=0.5 --batch_size=64 --num_epochs=200 --evaluate_every=1 --checkpoing_every=1 --num_checkpoints=30 --shuffle=True
~~~

## TODO
- [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template)를 참고하여 구조 바꾸기
- Embedding build하는 것 만들기 (Word2Vec, FastText, GloVe)
- 더 많은 모델 넣기
- feature extractor 넣기
- 몇가지 데이터셋에 대해 benchmark 확인
- evaluation function, loss function 더 많이 구현하고 자동화
- mode 추가 (infer)