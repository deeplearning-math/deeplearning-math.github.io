from fastai.vision.all import *
from fastai import *
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from efficientnet_pytorch import EfficientNet
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'#'1, 2, 3'
print("begin")

fold = 0
bs = 32

#load train data
path = Path('.')
f_get = partial(get_image_files,folders='train')

neg_files = Path(path/'train/train_contest/defect').ls()
pos_files = Path(path/'train/train_contest/good_all').ls()
test_files = Path(path/'test/test_contest/test').ls()
filenames,test_filenames, labels = [], [], []
bad_files = []
for f in neg_files:
    filenames.append(f.stem)
    bad_files.append(f.stem)
    labels.append(1)
for f in pos_files:
    filenames.append(f.stem)
    labels.append(0)
for f in test_files:
    test_filenames.append(f.stem)
# df_train = pd.DataFrame(list(zip(filenames,labels)),columns=['id','label'])
df_train = pd.DataFrame(labels,index=filenames,columns=['label'])
df_train.head() #train data

dblocks = DataBlock(blocks = (ImageBlock, CategoryBlock),
                    get_items = f_get,
                    splitter = RandomSplitter(seed=42),
                    get_y = parent_label,
                    item_tfms = Resize(224),
                    batch_tfms=[*aug_transforms(max_warp=0.),Normalize.from_stats(*imagenet_stats)]
                   )


# count = Counter(df_train['label']).values()
# class_weights = 1/np.array(list(count))
# dsets = dblocks.datasets(path)
# wgts = class_weights[list(df_train.loc[list(map(lambda x: x.stem,dsets.train.items))]['label'])]
# dls = dblocks.dataloaders(path,bs=bs,num_workers=16,dl_type=WeightedDL,wgts=wgts)
dls = dblocks.dataloaders(path,bs=bs,num_workers=16)


def auc_roc_metrics(preds,targs):
    return RocAucBinary()(preds.argmax(dim=-1),targs)

def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b7',num_classes=2)
    return net


learn=Learner(dls,get_net(),loss_func=LabelSmoothingCrossEntropy(),metrics=auc_roc_metrics).to_fp16()  
# learn.to_parallel()
learn.fit_one_cycle(5,1e-5) 
learn.save('effnet-b7-1')

df_test = pd.read_csv(path/'submission_sample.csv')
def get_test_items(df,path):
    files = [str(path)+'/test/test_contest/test/'+x+'.bmp' for x in df['id']]
    return files
test_dl = dls.test_dl(get_test_items(df_test,path))
# learn = cnn_learner(dls,resnet50,loss_func=LabelSmoothingCrossEntropy(),metrics=auc_roc_metrics,model_dir='/kaggle/working/res50-baseline').to_fp16()
preds = learn.get_preds(dl=test_dl)
result = {'id': [], 'defect_score': []}

for fname,score in zip(df_test['id'],preds[0][:,0].tolist()):
    result['id'].append(fname)
    result['defect_score'].append(score)
submission = pd.DataFrame(result)
submission.to_csv('submission-effnet-b7-1.csv', index=False) 



learn.unfreeze()                          
learn.fit_one_cycle(20,1e-5) 
                         
                         
learn.save('effnet-b7-2')

df_test = pd.read_csv(path/'submission_sample.csv')
def get_test_items(df,path):
    files = [str(path)+'/test/test_contest/test/'+x+'.bmp' for x in df['id']]
    return files
test_dl = dls.test_dl(get_test_items(df_test,path))
# learn = cnn_learner(dls,resnet50,loss_func=LabelSmoothingCrossEntropy(),metrics=auc_roc_metrics,model_dir='/kaggle/working/res50-baseline').to_fp16()
preds = learn.get_preds(dl=test_dl)
result = {'id': [], 'defect_score': []}

for fname,score in zip(df_test['id'],preds[0][:,0].tolist()):
    result['id'].append(fname)
    result['defect_score'].append(score)
submission = pd.DataFrame(result)
submission.to_csv('submission-effnet-b7-1.csv', index=False)                          

