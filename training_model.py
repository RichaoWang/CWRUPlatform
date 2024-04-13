#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc  : 训练模型的相关函数
"""

from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, LSTM, Dropout, GRU
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras import backend as K
from keras.layers import multiply, Input, Reshape
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf

from feature_extraction import feature_extraction


def training_with_MCNN_LSTMBK(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=8,
                              num_classes=10):
    """
    备份 MCNN_LSTM
    Args:
        X_train:
        y_train:
        X_valid:
        y_valid:
        X_test:
        y_test:
        batch_size:
        epochs:
        num_classes:

    Returns:

    """
    X_train, X_valid, X_test = X_train[:, :, np.newaxis], X_valid[:, :, np.newaxis], X_test[:, :,
                                                                                     np.newaxis]  # 添加一个新的维度
    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    input_seq = Input(shape=(2048,))
    X = Reshape((1, 2048))(input_seq)

    # encoder1
    ec1_layer1 = Conv1D(filters=80, kernel_size=20, strides=4,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec1_layer2 = Conv1D(filters=60, kernel_size=10, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec1_layer1)
    ec1_layer3 = Conv1D(filters=30, kernel_size=10, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec1_layer2)
    ec1_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec1_layer3)

    # encoder2
    ec2_layer1 = Conv1D(filters=80, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec2_layer2 = Conv1D(filters=70, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer1)
    ec2_layer3 = MaxPooling1D(pool_size=4, strides=None, padding='valid',
                              data_format='channels_first')(ec2_layer2)
    ec2_layer4 = Conv1D(filters=60, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer3)
    ec2_layer5 = Conv1D(filters=60, kernel_size=8, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer4)

    ec2_layer6 = Conv1D(filters=30, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer5)

    ec2_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec2_layer6)

    encoder = multiply([ec1_outputs, ec2_outputs])

    dc_layer1 = LSTM(60, return_sequences=True)(encoder)
    dc_layer1_1 = Dropout(0.3)(dc_layer1)
    dc_layer2 = LSTM(120)(dc_layer1_1)
    dc_layer3 = Dropout(0.3)(dc_layer2)
    dc_layer4 = Dense(10, activation='softmax')(dc_layer3)

    model = Model(input_seq, dc_layer4)

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()

    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=2, validation_data=(X_valid, y_valid),
                        shuffle=True, initial_epoch=0)

    score = model.evaluate(X_test, y_test, verbose=0)

    return model, history, score


def training_with_MCNN_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=40,
                            num_classes=10):
    X_train, X_valid, X_test = X_train[:, :, np.newaxis], X_valid[:, :, np.newaxis], X_test[:, :,
                                                                                     np.newaxis]  # 添加一个新的维度
    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    input_seq = Input(shape=(2048,))  # (None, x)
    X = Reshape((1, 2048))(input_seq)  # (None, 1, x)

    # encoder1
    ec1_layer1 = Conv1D(filters=50, kernel_size=20, strides=2, kernel_regularizer=l2(1e-4),
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec1_layer2 = Conv1D(filters=30, kernel_size=10, strides=2, kernel_regularizer=l2(1e-4),
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec1_layer1)
    ec1_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec1_layer2)

    # encoder2
    ec2_layer1 = Conv1D(filters=50, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first', kernel_regularizer=l2(1e-4))(X)
    ec2_layer2 = Conv1D(filters=40, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first', kernel_regularizer=l2(1e-4))(ec2_layer1)
    ec2_layer3 = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                              data_format='channels_first')(ec2_layer2)
    ec2_layer4 = Conv1D(filters=30, kernel_size=8, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first', kernel_regularizer=l2(1e-4))(ec2_layer3)
    ec2_layer5 = Conv1D(filters=30, kernel_size=8, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first', kernel_regularizer=l2(1e-4))(ec2_layer4)
    ec2_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec2_layer5)

    # encoder3
    ec3_layer1 = Conv1D(filters=50, kernel_size=8, strides=2,
                        padding='valid', activation='tanh', kernel_regularizer=l2(1e-4),
                        data_format='channels_first')(X)
    ec3_layer2 = Conv1D(filters=40, kernel_size=8, strides=2,
                        padding='valid', activation='tanh', kernel_regularizer=l2(1e-4),
                        data_format='channels_first')(ec3_layer1)
    ec3_layer3 = Conv1D(filters=30, kernel_size=6, strides=1,
                        padding='valid', activation='tanh', kernel_regularizer=l2(1e-4),
                        data_format='channels_first')(ec3_layer2)
    ec3_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec3_layer3)

    encoder = multiply([ec1_outputs, ec2_outputs, ec3_outputs])

    dc_layer1 = LSTM(60, return_sequences=True, kernel_regularizer=l2(1e-3))(encoder)
    dc_layer2 = LSTM(60)(dc_layer1)
    dc_layer3 = Dropout(0.5)(dc_layer2)
    dc_layer4 = Dense(10, activation='softmax')(dc_layer3)

    model = Model(input_seq, dc_layer4)

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    model.summary()

    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=2, validation_data=(X_valid, y_valid),
                        shuffle=True, initial_epoch=0)

    score = model.evaluate(X_test, y_test, verbose=0)

    return model, history, score


def training_with_1D_CNN(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=20, num_classes=10):
    '''
    使用 1D_CNN 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_valid, X_test = X_train[:, :, np.newaxis], X_valid[:, :, np.newaxis], X_test[:, :,
                                                                                     np.newaxis]  # 添加一个新的维度
    # 输入数据的维度
    input_shape = X_train.shape[1:]

    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    # 实例化一个Sequential
    model = Sequential()

    # 第一层卷积
    model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))
    # 从卷积到全连接需要展平
    model.add(Flatten())
    # 添加全连接层
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
    # 增加输出层，共num_classes个单元
    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

    # 编译模型
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 开始模型训练
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_valid, y_valid), shuffle=True)
    # 评估模型
    score = model.evaluate(X_test, y_test, verbose=0)

    return model, history, score


def training_with_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 LSTM 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_valid, X_test = X_train[:, np.newaxis, :], X_valid[:, np.newaxis, :], X_test[:, np.newaxis,
                                                                                     :]  # 添加一个新的维度
    # print(X_train.shape)  # (7000, 1, 2048)
    # 输入数据的维度
    input_shape = X_train.shape[1:]  # (1, 2048)

    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    model_LSTM = Sequential()
    # LSTM 第一层
    model_LSTM.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model_LSTM.add(Dropout(0.5))
    # LSTM 第二层
    model_LSTM.add(LSTM(128, return_sequences=True))
    model_LSTM.add(Dropout(0.5))
    # LSTM 第三层
    model_LSTM.add(LSTM(256))
    model_LSTM.add(Dropout(0.5))
    # Dense层
    model_LSTM.add(Dense(num_classes, activation='sigmoid'))

    # 编译模型
    model_LSTM.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_LSTM.summary()

    # 开始模型训练
    history = model_LSTM.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(X_valid, y_valid), shuffle=True)
    # 评估模型
    score = model_LSTM.evaluate(X_test, y_test, verbose=0)

    return model_LSTM, history, score


def training_with_GRU(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 GRU 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_valid, X_test = X_train[:, np.newaxis, :], X_valid[:, np.newaxis, :], X_test[:, np.newaxis,
                                                                                     :]  # 添加一个新的维度
    # 输入数据的维度
    input_shape = X_train.shape[1:]

    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    model_GRU = Sequential()
    model_GRU.add(GRU(64, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model_GRU.add(Dropout(0.5))
    model_GRU.add(GRU(128, activation='tanh'))
    model_GRU.add(Dropout(0.5))
    model_GRU.add(Dense(num_classes, activation='sigmoid'))

    # 编译模型
    model_GRU.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # 开始模型训练
    history = model_GRU.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(X_valid, y_valid), shuffle=True)
    # 评估模型
    score = model_GRU.evaluate(X_test, y_test, verbose=0)

    return model_GRU, history, score


def training_with_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test):
    '''
    使用 随机森林 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
            clf_rfc：训练完成的模型
            score：模型在验证集上的得分
            X_train_feature_extraction：将原数据进行了特征提取过的训练集
            X_test_feature_extraction：将原数据进行了特征提取过的测试集
    '''
    # 把训练集和验证集合并，全部用作训练集
    X_train = np.vstack((X_train, X_valid))
    y_train = np.vstack((y_train, y_valid))

    # 将one-hot编码了的标签解码（这里不需要one-hot编码）
    y_train = [np.argmax(item) for item in y_train]
    y_train = np.array(y_train)
    y_test = [np.argmax(item) for item in y_test]
    y_test = np.array(y_test)

    loader = np.empty(shape=[X_train.shape[0], 16])
    for i in range(X_train.shape[0]):
        loader[i] = feature_extraction(X_train[i])
    X_train_feature_extraction = loader

    loader = np.empty(shape=[X_test.shape[0], 16])
    for i in range(X_test.shape[0]):
        loader[i] = feature_extraction(X_test[i])
    X_test_feature_extraction = loader

    clf_rfc = RandomForestClassifier(n_estimators=17, max_depth=21, criterion='gini', min_samples_split=2,
                                     max_features=9, random_state=60)
    clf_rfc.fit(X_train_feature_extraction, y_train)
    score = clf_rfc.score(X_test_feature_extraction, y_test)
    return clf_rfc, score, X_train_feature_extraction, X_test_feature_extraction


if __name__ == '__main__':
    from data_preprocess import training_stage_prepro

    data_path = "datas/0HP/"
    signal_length = 2048
    signal_number = 1000  # 每个类别中要抽取的样本的数量
    normal = True  # 是否标准化
    rate = [0.6, 0.2, 0.2]  # 训练集、测试集、验证集划分比例
    cache_path = "./cache"
    modelname = "MCNN_LSTM"

    print("是否gpu: ", tf.test.is_gpu_available())


    def training(data_path, signal_length, signal_number, normal, rate, save_path, model_name):
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(data_path,
                                                                                                signal_length,
                                                                                                signal_number, normal,
                                                                                                rate,
                                                                                                enhance=False)
        # print(X_train.shape) # (7000, 2048)
        # print(y_train.shape) # (7000, 10)
        model, history, score = training_with_MCNN_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test)
        # training_with_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test)
        # model, history, score = training_with_1D_CNN(X_train, y_train, X_valid, y_valid, X_test, y_test)
        print(score)
        # model, history, score = training_with_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test)
        # print(history)
        # plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
        # plot_confusion_matrix(model, model_name, save_path, X_test, y_test)  # 绘制混淆矩阵
        # classification_report = brief_classification_report(model, model_name, X_test, y_test)  # 计算分类报告
        # plot_metrics(model, model_name, save_path, X_test, y_test)  # 绘制 召回率曲线和精确度曲线


    training(data_path=data_path, signal_length=signal_length, signal_number=signal_number,
             normal=True, rate=rate, save_path=cache_path, model_name=modelname)
