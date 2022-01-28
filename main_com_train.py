# -*- coding:utf-8 -*-
from charset_normalizer import from_bytes
from modified_deeplab_V3 import *
from PFB_measurement import Measurement
from random import shuffle, random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/train.txt",

                           "val_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/val.txt",

                           "test_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/test.txt",
                           
                           "label_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/raw_aug_gray_mask/",
                           
                           "image_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/raw_aug_rgb_img/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuhwan/yuhwan/checkpoint/Segmenation/V2/BoniRob/sample_images",

                           "save_checkpoint": "/yuhwan/yuhwan/checkpoint/Segmenation/V2/BoniRob/checkpoint",

                           "save_print": "/yuhwan/yuhwan/checkpoint/Segmenation/V2/BoniRob/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.9)
color_map = np.array([[255, 0, 0], [0, 0, 255], [0,0,0]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    no_img = img
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def test_func2(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    temp_img = img
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, temp_img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def false_dice_loss(y_true, y_pred):
    y_true = 1 - tf.cast(y_true, tf.float32)
    y_pred = 1 - tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def modified_dice_loss_object(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = (y_true * (1 - tf.math.sigmoid(y_pred)) * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))
    denominator = (y_true * tf.math.sigmoid(y_pred) * 0.9 * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + 1)
    loss = tf.math.divide(numerator, denominator)
    loss = tf.reduce_mean(loss)

    return loss

def modified_dice_loss_nonobject(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = ((1  - y_true) * tf.math.sigmoid(y_pred) * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))
    denominator = ((1  - y_true) * (1 - tf.math.sigmoid(y_pred)) * 0.9 * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + 1)
    loss = tf.math.divide(numerator, denominator)
    loss = tf.reduce_mean(loss)

    return loss

def cal_loss(model, images, labels, objectiness, class_im_plain, ignore_label):

    with tf.GradientTape() as tape:

        
        batch_labels = tf.reshape(labels, [-1,])
        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])

        # Dice for background
        background_indices = tf.squeeze(tf.where(tf.equal(batch_labels, 2)), -1)
        background_labels = tf.gather(batch_labels, background_indices)
        background_labels = tf.zeros_like(background_labels, dtype=tf.float32)
        background_logits = tf.gather(logits[:, 2], background_indices)
        loss2 = false_dice_loss(background_labels, background_logits) \
            + modified_dice_loss_nonobject(background_labels, background_logits)

        non_background_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 2)), -1)
        non_background_labels = tf.gather(batch_labels, non_background_indices)
        non_background_labels = tf.ones_like(non_background_labels, dtype=tf.float32)
        non_background_logits = tf.gather(logits[:, 2], non_background_indices)
        loss2 += true_dice_loss(non_background_labels, non_background_logits) \
            + modified_dice_loss_object(non_background_labels, non_background_logits)

        # Dice for Crop
        crop_indices = tf.squeeze(tf.where(tf.equal(batch_labels, 0)), -1)
        crop_labels = tf.gather(batch_labels, crop_indices)
        crop_labels = tf.ones_like(crop_labels, dtype=tf.float32)
        crop_logits = tf.gather(logits[:, 0], crop_indices)
        loss4 = true_dice_loss(crop_labels, crop_logits) \
            + modified_dice_loss_object(crop_labels, crop_logits)

        non_crop_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 0)), -1)
        non_crop_labels = tf.gather(batch_labels, non_crop_indices)
        non_crop_labels = tf.zeros_like(non_crop_labels, dtype=tf.float32)
        non_crop_logits = tf.gather(logits[:, 0], non_crop_indices)
        loss4 += false_dice_loss(non_crop_labels, non_crop_logits) \
            + modified_dice_loss_nonobject(non_crop_labels, non_crop_logits)
        
        # Dice for weed
        weed_indices = tf.squeeze(tf.where(tf.equal(batch_labels, 1)), -1)
        weed_labels = tf.gather(batch_labels, weed_indices)
        weed_labels = tf.ones_like(weed_labels, dtype=tf.float32)
        weed_logits = tf.gather(logits[:, 1], weed_indices)
        loss5 = true_dice_loss(weed_labels, weed_logits) \
            + modified_dice_loss_object(weed_labels, weed_logits)
        
        non_weed_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 1)), -1)
        non_weed_labels = tf.gather(batch_labels, non_weed_indices)
        non_weed_labels = tf.zeros_like(non_weed_labels, dtype=tf.float32)
        non_weed_logits = tf.gather(logits[:, 1], non_weed_indices)
        loss5 += false_dice_loss(non_weed_labels, non_weed_logits) \
            + modified_dice_loss_nonobject(non_weed_labels, non_weed_logits)

        loss = loss4 + loss5 + loss2

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    with tf.GradientTape() as tape:

        
        batch_labels = tf.reshape(labels, [-1,])
        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])

        # Softmax cross entropy --> crop, weed, background
        sigmoid_logits = tf.nn.sigmoid(logits)
        loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_labels, logits * sigmoid_logits)

        loss = loss1

    grads = tape.gradient(loss, model.trainable_variables)

    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()
    # 마지막 plain은 objecttines에 대한 True or False값 즉 (mask값이고), 라벨은 annotation 이미지임 (crop/weed)
    # 학습이미지에 대해 online augmentation을 진행--> 전처리로서 필터링을 하던지 해서 , 피사체에 대한 high frequency 성분을
    # 가지고오자
    model = DeepLabV3Plus(FLAGS.img_size, FLAGS.img_size, 34)
    out = model.get_layer("activation_decoder_2_upsample").output
    out = tf.keras.layers.Conv2D(FLAGS.total_classes, (1, 1), name='output_layer')(out)
    model = tf.keras.Model(inputs=model.input, outputs=out)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
           layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)
        # elif isinstance(layer, tf.keras.layers.Conv2D):
        #     layer.kernel_initializer = tf.keras.initializers.HeNormal()

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        val_img_dataset = [FLAGS.image_path + data for data in val_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)
                batch_labels = np.squeeze(batch_labels, -1)


                
                class_imbal_labels = batch_labels
                class_imbal_labels_buf = 0.
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab
                class_imbal_labels_buf /= 2
                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes-1]
                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                class_im_plain = np.where(batch_labels == 0, class_imbal_labels_buf[0], batch_labels)
                class_im_plain = np.where(batch_labels == 1, class_imbal_labels_buf[1], batch_labels)
                #a = np.reshape(class_im_plain, [FLAGS.batch_size*FLAGS.img_size*FLAGS.img_size, ])
                #a = np.array(a, dtype=np.int32)
                #a = np.bincount(a, minlength=3)
                objectiness = np.where(batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것

                loss = cal_loss(model, batch_images, batch_labels, objectiness, class_im_plain, 2)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    output = run_model(model, batch_images, False)
                    att_crop_output = tf.nn.sigmoid(output[:, :, :, 0])
                    att_weed_output = tf.nn.sigmoid(output[:, :, :, 1])
                    att_back_output = tf.nn.sigmoid(output[:, :, :, 2])
                    temp = tf.nn.softmax(output[:, :, :, 0:2], -1)
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i], tf.int32).numpy()
                        att_crop = att_crop_output[i].numpy()
                        att_weed = att_weed_output[i].numpy()
                        att_back = att_back_output[i].numpy()
                        image = output[i].numpy()
                        image[:, :, 0] = image[:, :, 0] * att_crop
                        image[:, :, 1] = image[:, :, 1] * att_weed
                        image[:, :, 2] = image[:, :, 2] * att_back
                        image = tf.nn.softmax(image, -1)
                        image = tf.cast(tf.argmax(image, -1), tf.int32).numpy()
                        pred_mask_color = color_map[image]
                        label_mask_color = color_map[label]

                        temp_img = np.concatenate((image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]), -1)
                        image = np.concatenate((image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]), -1)
                        pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), print_images[i], image)
                        pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping /= 255.
 
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_warping_predict.png", pred_mask_warping)
                    

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    output = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????
                    temp = tf.nn.softmax(output[0, :, :, 0:2], -1)
                    att_crop = tf.nn.sigmoid(output[0, :, :, 0])
                    att_weed = tf.nn.sigmoid(output[0, :, :, 1])
                    att_back = tf.nn.sigmoid(output[0, :, :, 2])
                    image = output[0].numpy()
                    image[:, :, 0] = image[:, :, 0] * att_crop
                    image[:, :, 1] = image[:, :, 1] * att_weed
                    image[:, :, 2] = image[:, :, 2] * att_back
                    image = tf.nn.softmax(image, -1)
                    image = tf.cast(tf.argmax(image, -1), tf.int32).numpy()

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)


                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_, _, _, _, _ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("=================================================================================================================================================")
            print("Epoch: %3d, train mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), train F1_score = %.4f, train sensitivity = %.4f, train TDR = %.4f" % (epoch, miou / len(train_img_dataset),
                                                                                                                                                 crop_iou / len(train_img_dataset),
                                                                                                                                                 weed_iou / len(train_img_dataset),
                                                                                                                                                  f1_score / len(train_img_dataset),
                                                                                                                                                  sensitivity / len(train_img_dataset),
                                                                                                                                                  tdr / len(train_img_dataset)))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (miou / len(train_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(train_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(train_img_dataset)))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score / len(train_img_dataset)))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(train_img_dataset)))
            output_text.write(", train TDR: ")
            output_text.write("%.4f" % (tdr / len(train_img_dataset)))
            output_text.write("\n")

            val_iter = iter(val_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    output = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????
                    temp = tf.nn.softmax(output[0, :, :, 0:2], -1)
                    att_crop = tf.nn.sigmoid(output[0, :, :, 0])
                    att_weed = tf.nn.sigmoid(output[0, :, :, 1])
                    att_back = tf.nn.sigmoid(output[0, :, :, 2])
                    image = output[0].numpy()
                    image[:, :, 0] = image[:, :, 0] * att_crop
                    image[:, :, 1] = image[:, :, 1] * att_weed
                    image[:, :, 2] = image[:, :, 2] * att_back
                    image = tf.nn.softmax(image, -1)
                    image = tf.cast(tf.argmax(image, -1), tf.int32).numpy()

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_, _, _, _, _ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, val mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), val F1_score = %.4f, val sensitivity = %.4f, val TDR = %.4f" % (epoch, miou / len(val_img_dataset),
                                                                                                                                         crop_iou / len(val_img_dataset),
                                                                                                                                         weed_iou / len(val_img_dataset),
                                                                                                                                         f1_score / len(val_img_dataset),
                                                                                                                                         sensitivity / len(val_img_dataset),
                                                                                                                                         tdr / len(val_img_dataset)))
            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (miou / len(val_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(val_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(val_img_dataset)))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score / len(val_img_dataset)))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(val_img_dataset)))
            output_text.write(", val TDR: ")
            output_text.write("%.4f" % (tdr / len(val_img_dataset)))
            output_text.write("\n")

            test_iter = iter(test_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    output = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????
                    temp = tf.nn.softmax(output[0, :, :, 0:2], -1)
                    att_crop = tf.nn.sigmoid(output[0, :, :, 0])
                    att_weed = tf.nn.sigmoid(output[0, :, :, 1])
                    att_back = tf.nn.sigmoid(output[0, :, :, 2])
                    image = output[0].numpy()
                    image[:, :, 0] = image[:, :, 0] * att_crop
                    image[:, :, 1] = image[:, :, 1] * att_weed
                    image[:, :, 2] = image[:, :, 2] * att_back
                    image = tf.nn.softmax(image, -1)
                    image = tf.cast(tf.argmax(image, -1), tf.int32).numpy()

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_, _, _, _, _ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=image,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity = %.4f, test TDR = %.4f" % (epoch, miou / len(test_img_dataset),
                                                                                                                                             crop_iou / len(test_img_dataset),
                                                                                                                                             weed_iou / len(test_img_dataset),
                                                                                                                                             f1_score / len(test_img_dataset),
                                                                                                                                             sensitivity / len(test_img_dataset),
                                                                                                                                             tdr / len(test_img_dataset)))
            print("=================================================================================================================================================")
            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (miou / len(test_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(test_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(test_img_dataset)))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score / len(test_img_dataset)))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(test_img_dataset)))
            output_text.write(", test TDR: ")
            output_text.write("%.4f" % (tdr / len(test_img_dataset)))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)

    # else:
    #     test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

    #     test_img_dataset = [FLAGS.image_path + data for data in test_list]
    #     test_lab_dataset = [FLAGS.label_path + data for data in test_list]

    #     test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
    #     test_ge = test_ge.map(test_func2)
    #     test_ge = test_ge.batch(1)
    #     test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

    #     test_iter = iter(test_ge)
    #     miou = 0.
    #     f1_score = 0.
    #     tdr = 0.
    #     sensitivity = 0.
    #     crop_iou = 0.
    #     weed_iou = 0.
    #     pre_ = 0.
    #     TP_, TN_, FP_, FN_ = 0, 0, 0, 0
    #     for i in range(len(test_img_dataset)):
    #         batch_images, nomral_img, batch_labels = next(test_iter)
    #         batch_labels = tf.squeeze(batch_labels, -1)
    #         for j in range(1):
    #             batch_image = tf.expand_dims(batch_images[j], 0)
    #             logits = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????
    #             object_predict = tf.nn.sigmoid(logits[0, :, :, 1])
    #             predict = tf.nn.sigmoid(logits[0, :, :, 0:1])
    #             predict = np.where(predict.numpy() >= 0.5, 1, 0)
    #             predict_temp = predict
    #             image = predict
    #             object_predict_predict = np.where(object_predict.numpy() >= 0.5, 1, 2)
    #             onject_predict_axis = np.where(object_predict_predict==2)   # 2 ???漺???? ?ִ? ?ุ ?????? ??
    #             predict_temp[onject_predict_axis] = 2

    #             #batch_image = tf.expand_dims(batch_images[j], 0)
    #             #predict = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????
    #             #predict = tf.nn.sigmoid(predict[0, :, :, 0:1])
    #             #predict = np.where(predict.numpy() >= 0.5, 1, 0)

    #             batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
    #             batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
    #             batch_label = np.where(batch_label == 255, 0, batch_label)
    #             batch_label = np.where(batch_label == 128, 1, batch_label)
    #             ignore_label_axis = np.where(batch_label==2)   # ?????? x,y axis?? ????!
    #             predict[ignore_label_axis] = 2

    #             predict_temp1 = predict_temp
    #             batch_label1 = batch_label
    #             miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_temp1,
    #                                 label=batch_label1, 
    #                                 shape=[FLAGS.img_size*FLAGS.img_size, ], 
    #                                 total_classes=FLAGS.total_classes).MIOU()
    #             predict_temp2 = predict_temp
    #             batch_label2 = batch_label
    #             f1_score_, recall_, TP, TN, FP, FN = Measurement(predict=predict_temp2,
    #                                     label=batch_label2,
    #                                     shape=[FLAGS.img_size*FLAGS.img_size, ],
    #                                     total_classes=FLAGS.total_classes).F1_score_and_recall()
    #             predict_temp3 = predict_temp
    #             batch_label3 = batch_label
    #             tdr_ = Measurement(predict=predict_temp3,
    #                                     label=batch_label3,
    #                                     shape=[FLAGS.img_size*FLAGS.img_size, ],
    #                                     total_classes=FLAGS.total_classes).TDR()
    #             predict_temp4 = predict_temp
    #             batch_label4 = batch_label
    #             output_confusion, temp_output_3D = Measurement(predict=predict_temp4,
    #                                     label=batch_label4,
    #                                     shape=[FLAGS.img_size*FLAGS.img_size, ],
    #                                     total_classes=FLAGS.total_classes).show_confusion()
    #             output_confusion_ = output_confusion

    #             pred_mask_color = color_map[predict_temp]  # ?????׸?ó?? ?Ұ?!
    #             pred_mask_color = np.squeeze(pred_mask_color, 2)
    #             batch_label = np.expand_dims(batch_label, -1)
    #             batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
    #             label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
    #             label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
    #             label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

    #             temp_img = np.concatenate((predict_temp, predict_temp, predict_temp), -1)
    #             image = np.concatenate((image, image, image), -1)
    #             pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), nomral_img[j], image)
    #             pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
    #             pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
    #             pred_mask_warping /= 255.

    #             output_confusion_warp = np.where(temp_output_3D == np.array([0,0,0], dtype=np.uint8), nomral_img[j], temp_output_3D)
    #             output_confusion_warp = np.where(temp_output_3D == np.array([10,10,10], dtype=np.uint8), np.array([255, 127, 0], dtype=np.uint8), output_confusion_warp)   # TN    - ??Ȳ??
    #             output_confusion_warp = np.where(temp_output_3D == np.array([20,20,20], dtype=np.uint8), np.array([128, 128, 128], dtype=np.uint8), output_confusion_warp)    # TP    - ȸ??
    #             output_confusion_warp = np.where(temp_output_3D == np.array([30,30,30], dtype=np.uint8), np.array([139, 0, 255], dtype=np.uint8), output_confusion_warp)    # FN    - ??????
    #             output_confusion_warp = np.where(temp_output_3D == np.array([40,40,40], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8), output_confusion_warp)    # FP    - ??????
    #             output_confusion_warp = np.array(output_confusion_warp, dtype=np.int32)
    #             output_confusion_warp = output_confusion_warp / 255

    #             name = test_img_dataset[i].split("/")[-1].split(".")[0]
    #             plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
    #             plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)
    #             plt.imsave(FLAGS.test_images + "/" + name + "_predict_warp.png", pred_mask_warping)
    #             plt.imsave(FLAGS.test_images + "/" + name + "_output_confusion.png", output_confusion)
    #             plt.imsave(FLAGS.test_images + "/" + name + "output_confusion_warp.png", output_confusion_warp)

    #             miou += miou_
    #             f1_score += f1_score_
    #             sensitivity += recall_
    #             tdr += tdr_
    #             crop_iou += crop_iou_
    #             weed_iou += weed_iou_
    #             TP_ += TP   # ?? ?κ??? ?̹????? ǥ???ؾ??Ѵ?
    #             TN_ += TN   # ?? ?κ??? ?̹????? ǥ???ؾ??Ѵ?
    #             FP_ += FP   # ?? ?κ??? ?̹????? ǥ???ؾ??Ѵ?
    #             FN_ += FN   # ?? ?κ??? ?̹????? ǥ???ؾ??Ѵ?


    #     print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity = %.4f, test TDR = %.4f" % (miou / len(test_img_dataset),
    #                                                                                                                                         crop_iou / len(test_img_dataset),
    #                                                                                                                                         weed_iou / len(test_img_dataset),
    #                                                                                                                                         f1_score / len(test_img_dataset),
    #                                                                                                                                         sensitivity / len(test_img_dataset),
    #                                                                                                                                         tdr / len(test_img_dataset)))
    #     #print(pre_ / len(test_img_dataset))
    #     TP_FP = (TP_ + FP_) + 1e-7

    #     TP_FN = (TP_ + FN_) + 1e-7

    #     out = np.zeros((1))
    #     Precision = np.divide(TP_, TP_FP)
    #     Recall = np.divide(TP_, TP_FN)

    #     Pre_Re = (Precision + Recall) + 1e-7

    #     F1_score = np.divide(2. * (Precision * Recall), Pre_Re)
    #     print(F1_score, Recall)

if __name__ == "__main__":
    main()
