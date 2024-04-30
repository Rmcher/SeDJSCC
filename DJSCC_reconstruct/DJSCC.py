import tensorflow as tf
from keras.layers import Input, Lambda
from keras import Model
from keras.optimizers import Adam
import numpy as np
import argparse
from dataset import dataset_cifar10
from DJSCC_reconstruct.channel import Channel
from DJSCC_reconstruct.model import Semantic_Encoder, Semantic_Decoder
import os
import json
import random as python_random

def reset_random_seeds(seed_value=41):
   tf.random.set_seed(seed_value)
   np.random.seed(seed_value)
   python_random.seed(seed_value)

def train(args, model):
    epoch_list = []
    loss_list = []
    val_loss_list = []
    min_loss = 10 ** 8

    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)

    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_train) + '_bs' + str(args.batch_size)+'_lr'+str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'

    if args.channel_type == 'awgn':
        (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(args.snr_train)
    elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
        (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(args.snr_train)
    train_step = (train_nums // args.batch_size if train_nums % args.batch_size == 0 else train_nums // args.batch_size + 1)
    valid_step = (test_nums // args.batch_size if test_nums % args.batch_size == 0 else test_nums // args.batch_size + 1)

    for epoch in range(0, args.epochs):
        if args.channel_type == 'awgn':
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(args.snr_train)
        elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(args.snr_train)
        
        train_ds = train_ds.shuffle(buffer_size=train_nums)
        train_ds = train_ds.batch(args.batch_size)
        test_ds = test_ds.batch(args.batch_size)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_step = (train_nums//args.batch_size if train_nums%args.batch_size==0 else train_nums//args.batch_size+1)
        valid_step = (test_nums//args.batch_size if test_nums%args.batch_size==0 else test_nums//args.batch_size+1)
        h = model.fit(train_ds, epochs=1, steps_per_epoch=train_step, validation_data=test_ds, validation_steps=valid_step)
        his = h.history
        loss = his.get('loss')[0]
        val_loss = his.get('val_loss')[0]

        if val_loss < min_loss:
            min_loss = val_loss
            model.save_weights(model_path)
            print(f'Epoch: {epoch + 1}, loss: {loss}, val_loss: {val_loss} - Model Saved')
        else:
            print(f'Epoch: {epoch + 1}, loss: {loss}, val_loss: {val_loss}')
        
        epoch_list.append(epoch)
        loss_list.append(loss)
        val_loss_list.append(val_loss)

        with open(args.loss_dir + filename + '.json', mode='w') as f:
            json.dump({'epoch': epoch_list, 'loss': loss_list, 'val_loss': val_loss_list}, f)

def eval(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_eval) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    snr_list = []
    mse_list = []
    psnr_list = []
    for snrdb in range(0, 21):
        imse = []
        # test 10 times each snr
        for i in range(0, 10):
            if args.channel_type == 'awgn':
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snrdb)
            elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(snrdb)
            test_ds = test_ds.shuffle(buffer_size=test_nums)
            test_ds = test_ds.batch(args.batch_size)
            mse = model.evaluate(test_ds)
            imse.append(mse)
        mse = np.mean(imse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        snr_list.append(snrdb)
        mse_list.append(mse)
        psnr_list.append(psnr)
        with open(args.eval_dir + filename + '.json', mode='w') as f:
            json.dump({'snr': snr_list, 'mse': mse_list, 'psnr': psnr_list}, f)

def main(args):
    # construct encoder-decoder model
    input_imgs = Input(shape=(32, 32, 3))
    input_snrdb = Input(shape=(1,))
    input_h_real = Input(shape=(1,))
    input_h_imag = Input(shape=(1,))
    imgs_norm = Lambda(lambda x: x / 255, name='normal')(input_imgs)

    channel_input = Semantic_Encoder(imgs_norm, args.transmit_channel_num)

    if args.channel_type == 'awgn':
        rv = Channel(channel_type='awgn')(channel_input, input_snrdb)
    elif args.channel_type == 'slow_fading':
        rv = Channel(channel_type='slow_fading')(channel_input, input_snrdb, input_h_real, input_h_imag)
    elif args.channel_type == 'slow_fading_eq':
        rv = Channel(channel_type='slow_fading_eq')(channel_input, input_snrdb, input_h_real, input_h_imag)
    
    normed_imgs = Semantic_Decoder(rv)
    imgs_rec = Lambda(lambda x: x * 255, name='denormal')(normed_imgs)

    if args.channel_type == 'awgn':
        model = Model(inputs=[input_imgs, input_snrdb], outputs=imgs_rec)
    elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
        model = Model(inputs=[input_imgs, input_snrdb, input_h_real, input_h_imag], outputs=imgs_rec)
    model.compile(Adam(args.learning_rate), 'mse')
    model.summary()

    if args.task == 'train':
        train(args, model)
    elif args.task == 'eval':
        eval(args, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help='train/eval')
    parser.add_argument("-c", '--channel_type', help="awgn/slow_fading", default="awgn")
    parser.add_argument("-m", '--model_dir', help="dir path for model saving", default='/home/wbh/JSCC/DJSCC_reconstruct/model/')
    parser.add_argument("-lm", '--load_model_path', help="model path for reloading")
    parser.add_argument("-b", "--batch_size", help="batch size", default=128, type=int)
    parser.add_argument("-e", "--epochs", help="epochs", default=500, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", default=0.0001, type=float)
    parser.add_argument("-cn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=48, type=int)
    parser.add_argument("-snr", "--snr_train", help="snr for training", default=20, type=int)
    parser.add_argument("-l", "--loss_dir", help="loss_dir for training", default='/home/wbh/JSCC/DJSCC_reconstruct/loss/')
    parser.add_argument("-ev", "--eval_dir", help="eval_dir", default='/home/wbh/JSCC/DJSCC_reconstruct/eval/')
    global args
    args = parser.parse_args()
    print("---------------------------------------")
    print("Parameters in the command argument for current task:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("---------------------------------------")

    reset_random_seeds()
    main(args)

# cmd
# python /home/wbh/JSCC/DJSCC_reconstruct/DJSCC.py train
# python /home/wbh/JSCC/DJSCC_reconstruct/DJSCC.py eval