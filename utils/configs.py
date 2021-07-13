from easydict import EasyDict as edict

config = edict()

config.model = edict()
config.model.ckp_path_high = 'checkpoint/highlayer/'
config.model.ckp_path_bot = 'checkpoint/botlayer/'
config.model.ckp_path_ft = 'checkpoint/ftlayer/'

config.model.tfrecord = 'dataset/tfrecord/'
config.model.tfrecord_dual = 'dataset/tfrecord/dual_'
config.model.tfrecord_ft = 'dataset/tfrecord/ft_'

config.model.loss_model = 'vgg_16'
config.model.loss_vgg = 'loss/pretrained/vgg16.npy'

config.data = edict()
config.data.hdr_path = 'dataset/train/hdr/'
config.data.sdr_path = 'dataset/train/sdr/'

config.data.patch_size_h = 512
config.data.patch_size_w = 512  # bot patch size is 2**(-lev) times of high
config.data.patch_size_ft_h = 256  # smaller patch size for fine_tuning due to limited GPU memory
config.data.patch_size_ft_w = 256
config.data.patch_ratio_x = 0.1
config.data.patch_ratio_y = 0.6
config.data.patch_per_img = 16  # 15 + 1

config.data.appendix_hdr = 'hdr'
config.data.appendix_sdr = 'jpg'

config.train = edict()
config.train.batch_size_high = 4  # OMM with value 8
config.train.batch_size_bot = 32
config.train.batch_size_ft = 8

config.train.train_set_size = 122  # 1700
config.train.total_imgs = config.data.patch_per_img * config.train.train_set_size
config.train.batchnum_high = round(config.train.total_imgs/config.train.batch_size_high)
config.train.batchnum_bot = round(config.train.total_imgs/config.train.batch_size_bot)
config.train.batchnum_ft = round(config.train.total_imgs/config.train.batch_size_ft)


config.test = edict()
config.test.tfrecord_test = 'dataset/tfrecord/test_'
config.test.result = 'result/'
config.test.hdr_path = 'dataset/test/'
# config.test.loss_vgg = 'loss/pretrained/vgg16.npy'