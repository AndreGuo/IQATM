import threading
import os, sys

'''Parameters to modify'''
# phase = 'high'
# phase = 'bot'
phase = 'ft'
'''===================='''


if phase == 'high':
    tfbd_cfg = 'tensorboard --logdir=' + 'checkpoint/highlayer/' + ' --port=8889'
elif phase == 'bot':
    tfbd_cfg = 'tensorboard --logdir=' + 'checkpoint/botlayer/' + ' --port=8890'
elif phase == 'ft':
    tfbd_cfg = 'tensorboard --logdir=' + 'checkpoint/ftlayer/' + ' --port=8891'
else:
    sys.exit('Please enter the right phase name!')

def launchTensorBoard():
    os.system(tfbd_cfg)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()