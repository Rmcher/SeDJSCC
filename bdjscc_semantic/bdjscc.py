from util_channel import Channel
from tensorflow import keras
from keras.layers import Input, Lambda
from keras import Model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import argparse
from dataset import dataset_cifar10
import os
import json
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from util_module import Modified_Basic_Encoder, Modified_Basic_Decoder
from keras.callbacks import ModelCheckpoint
import random as python_random

AUTOTUNE = tf.data.experimental.AUTOTUNE

def reset_random_seeds(seed_value=43):
   tf.random.set_seed(seed_value)
   np.random.seed(seed_value)
   python_random.seed(seed_value)

def train(args, model):
    epoch_list = []
    loss_list = []
    val_loss_list = []
    accuracy_list = []
    val_accuracy_list = []
    min_loss = 10 ** 8

    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)

    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_train) + '_bs' + str(args.batch_size)+'_lr'+str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'

    # 计算每10个epoch中的总batch数
    if args.channel_type == 'awgn':
        (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(args.snr_train)
    elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
        (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(args.snr_train)
    train_step = (train_nums // args.batch_size if train_nums % args.batch_size == 0 else train_nums // args.batch_size + 1)
    valid_step = (test_nums // args.batch_size if test_nums % args.batch_size == 0 else test_nums // args.batch_size + 1)

    save_freq = train_step * 10  # 每10个epochs保存一次

    checkpoint = ModelCheckpoint(
        filepath=model_path, 
        save_weights_only=True, 
        save_best_only=False,  # 设置为False以确保每10个epoch保存一次，无视性能
        monitor='val_loss', 
        mode='min', 
        save_freq=save_freq,  # 指定保存频率
        verbose=1
    )

    for epoch in range(0, args.epochs):
        if args.channel_type == 'awgn':
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(args.snr_train)
        elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(args.snr_train)
        
        train_ds = train_ds.shuffle(buffer_size=train_nums)
        train_ds = train_ds.batch(args.batch_size)
        test_ds = test_ds.batch(args.batch_size)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

        history = model.fit(train_ds, epochs=1, steps_per_epoch=train_step, validation_data=test_ds, validation_steps=valid_step, callbacks=[checkpoint])
        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        accuracy = history.history['accuracy'][0]  # 修改为 'accuracy'
        val_accuracy = history.history['val_accuracy'][0]  # 修改为 'val_accuracy'

        if val_loss < min_loss:
            min_loss = val_loss
            model.save_weights(model_path)
            print(f'Epoch: {epoch + 1}, Loss: {loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Val Accuracy: {val_accuracy} - Model Saved')
        else:
            print(f'Epoch: {epoch + 1}, Loss: {loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Val Accuracy: {val_accuracy}')

        epoch_list.append(epoch)
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)
        val_accuracy_list.append(val_accuracy)

        with open(args.loss_dir + filename + '.json', mode = 'w') as f:
            json.dump({'epoch': epoch_list, 'loss': loss_list, 'val_loss': val_loss_list, 'accuracy': accuracy_list, 'val_accuracy': val_accuracy_list}, f)

def eval_mismatch(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_train) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    snr_list = []
    accuracy_list = []
    for snrdb in range(0, 21):
        accuracies = []
        # test 10 times each snr
        for i in range(0, 4):
            if args.channel_type == 'awgn':
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snrdb)
            elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
                (_, _), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr_and_h(snrdb)
            test_ds = test_ds.shuffle(buffer_size=test_nums)
            test_ds = test_ds.batch(args.batch_size)
            _, accuracy = model.evaluate(test_ds)
            accuracies.append(accuracy)
        mean_accuracy = np.mean(accuracies)
        snr_list.append(snrdb)
        accuracy_list.append(mean_accuracy)
        with open(args.eval_dir + filename + '.json', mode='w') as f:
            json.dump({'snr': snr_list, 'accuracy': accuracy_list}, f)


def main(args):
    # construct encoder-decoder model
    input_imgs = Input(shape=(32, 32, 3))
    input_snrdb = Input(shape=(1,))
    input_h_real = Input(shape=(1,))
    input_h_imag = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)

    channel_input = Modified_Basic_Encoder(normal_imgs, args.transmit_channel_num)

    if args.channel_type == 'awgn':
        channel_output = Channel(channel_type='awgn')(channel_input, input_snrdb)
    elif args.channel_type == 'slow_fading':
        channel_output = Channel(channel_type='slow_fading')(channel_input, input_snrdb, input_h_real, input_h_imag)
    elif args.channel_type == 'slow_fading_eq':
        channel_output = Channel(channel_type='slow_fading_eq')(channel_input, input_snrdb, input_h_real, input_h_imag)
    
    prediction = Modified_Basic_Decoder(channel_output)
    
    if args.channel_type == 'awgn':
        model = Model(inputs=[input_imgs, input_snrdb], outputs=prediction)
    elif args.channel_type == 'slow_fading' or args.channel_type == 'slow_fading_eq':
        model = Model(inputs=[input_imgs, input_snrdb, input_h_real, input_h_imag], outputs=prediction)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 用于分类
              metrics=['accuracy'])
    model.summary()

    if args.command == 'train':
        train(args, model)
    elif args.command == 'eval_mismatch':
        eval_mismatch(args, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/eval_mismatch/train_mix')
    parser.add_argument("-ct", '--channel_type', help="awgn/slow_fading/slow_fading_eq", default="awgn")
    parser.add_argument("-md", '--model_dir', help="dir for model", default='/home/wbh/JSCC/bdjscc_semantic/model/')
    parser.add_argument("-lmp", '--load_model_path', help="model path for loading")
    parser.add_argument("-bs", "--batch_size", help="Batch size for training", default=128, type=int)
    parser.add_argument("-e", "--epochs", help="epochs for training", default=90, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=48,
                        type=int)
    parser.add_argument("-snr_train", "--snr_train", help="snr for training", default=6, type=int)
    parser.add_argument("-ldd", "--loss_dir", help="loss_dir for training", default='/home/wbh/JSCC/bdjscc_semantic/loss/')
    parser.add_argument("-ed", "--eval_dir", help="eval_dir", default='/home/wbh/JSCC/bdjscc_semantic/eval/')
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")

    reset_random_seeds()
    main(args)

# command
# python /home/wbh/JSCC/bdjscc_semantic/bdjscc.py train
# python /home/wbh/JSCC/bdjscc_semantic/bdjscc.py eval_mismatch
