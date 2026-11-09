"""合成画像を使ったCPU上の学習・保存・再開の統合検証。

uv run --no-project --index https://download.pytorch.org/whl/cpu \
    --with ultralytics==8.4.0 --with torch==2.9.1+cpu \
    --with torchvision==0.24.1+cpu --with tqdm python tests/verify_training_cell.py

公式の事前学習重みは取得せず、ランダム初期化した実YOLO26nを使います。
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import csv
import json
import tempfile
from pathlib import Path

import torch
torch.set_num_threads(1)
from PIL import Image
from ultralytics import YOLO
import ultralytics.data.utils as data_utils
# 図用フォントの取得だけ無効化。モデル・学習・検証・チェックポイント処理は実物を使います。
data_utils.check_font=lambda *a,**kw: None

notebook=json.loads((Path(__file__).resolve().parents[1] / 'yolo_person_finetuning.ipynb').read_text())
prepare=''.join(next(c for c in notebook['cells'] if c['id']=='dataset-prepare')['source'])
prepare=prepare.removesuffix('\nDATA_YAML = prepare_dataset(DATA_DIR, OUTPUT_DIR)\n')
source=''.join(next(c for c in notebook['cells'] if c['id']=='training')['source'])
source=source[:source.index('\nTRAINED_WEIGHTS = run_training(')]
ns={}
exec(prepare,ns)
exec(source,ns)
train=ns['run_training']

with tempfile.TemporaryDirectory(prefix='training_cell_test_') as temporary:
 root=Path(temporary)
 dataset=root/'data'
 for split in ('train','val'):
  for kind in ('images','labels'):
   (dataset/kind/split).mkdir(parents=True)
  for index in range(2):
   Image.new('RGB',(64,64),(30+index*50,60,90)).save(dataset/'images'/split/f'{index}.jpg')
   (dataset/'labels'/split/f'{index}.txt').write_text('0 0.5 0.5 0.6 0.8\n1 0.5 0.25 0.2 0.2\n')
 (dataset/'.build_config.json').write_text('{"synthetic":true}')
 (dataset/'preparation_report.json').write_text('{"synthetic":true}')
 yaml_path=dataset/'data.yaml'
 yaml_path.write_text(f'path: {json.dumps(str(dataset))}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: head\n')
 pretrained=root/'initial.pt'
 YOLO('yolo26n.yaml').save(pretrained)
 args=(yaml_path,pretrained,root/'drive_runs',root/'local_runs','resume_test',2,64,2,0)
 kw=dict(device='cpu',amp=False,plots=False,require_cuda=False)

 original_add=YOLO.add_callback
 def interrupted_add(self,event,callback):
  if event=='on_model_save':
   def interrupted(trainer):
    callback(trainer)
    raise RuntimeError('SIMULATED_SESSION_INTERRUPTION')
   return original_add(self,event,interrupted)
  return original_add(self,event,callback)
 YOLO.add_callback=interrupted_add
 try:
  train(*args,**kw)
 except RuntimeError as exc:
  assert str(exc)=='SIMULATED_SESSION_INTERRUPTION', str(exc)
 else:
  raise AssertionError('Interruption was not reached')
 finally:
  YOLO.add_callback=original_add
 run=root/'drive_runs'/'resume_test'
 assert json.loads((run/'latest.json').read_text())['completed_epochs']==1
 first=run/'epochs/epoch_000001/checkpoint.pt'
 ckpt=torch.load(first,map_location='cpu',weights_only=False)
 assert ckpt['epoch']==0 and ckpt['optimizer'] is not None and ckpt['ema'].nc==2
 del ckpt
 # Colabローカルが失われた状態を、別パスへの退避で再現する。
 (root/'local_runs').rename(root/'previous_session')

 # 2エポック目のDriveコピー失敗時、latestが1のままであることを確認する。
 original_copy=ns['copy_verified']
 def fail_copy(src,dst):
  if 'epoch_000002' in str(dst) and Path(dst).name=='best.pt':
   raise IOError('SIMULATED_DRIVE_FAILURE')
  return original_copy(src,dst)
 ns['copy_verified']=fail_copy
 try:
  train(*args,**kw)
 except IOError as exc:
  assert str(exc)=='SIMULATED_DRIVE_FAILURE', str(exc)
 else:
  raise AssertionError('Drive failure was not reached')
 finally:
  ns['copy_verified']=original_copy
 assert json.loads((run/'latest.json').read_text())['completed_epochs']==1

 result=train(*args,**kw)
 assert result.is_file()
 assert json.loads((run/'latest.json').read_text())['completed_epochs']==2
 second=run/'epochs/epoch_000002/checkpoint.pt'
 ckpt=torch.load(second,map_location='cpu',weights_only=False)
 assert ckpt['epoch']==1 and ckpt['optimizer'] is not None and ckpt['ema'].nc==2
 del ckpt
 rows=list(csv.DictReader((run/'epochs/epoch_000002/results.csv').open()))
 assert [int(row['epoch']) for row in rows]==[1,2]
 assert (run/'final/completed.json').is_file()
 # 完了済みの場合はtrainを呼ばない。
 original_train=YOLO.train
 def unexpected_train(*a,**kw):
  raise AssertionError('Completed run trained again')
 YOLO.train=unexpected_train
 try:
  assert train(*args,**kw)==second
 finally:
  YOLO.train=original_train
 # 既存RUN_NAMEでの設定変更を拒否する。
 try:
  train(*args[:5],3,*args[6:],**kw)
 except RuntimeError as exc:
  assert '別設定' in str(exc)
 else:
  raise AssertionError('Changed configuration accepted')
 # 保存済みファイルの破損を検出する。
 with second.open('ab') as stream:
  stream.write(b'corrupt')
 try:
  train(*args,**kw)
 except IOError as exc:
  assert '検証に失敗' in str(exc)
 else:
  raise AssertionError('Corrupt snapshot accepted')
print('PASS: 実YOLOの2クラス学習・エポック保存・セッション中断後の再開・Drive失敗時の再開先保持・CSV復元・完了時スキップ・設定変更拒否・破損検出')
