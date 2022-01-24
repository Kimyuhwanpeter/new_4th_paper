# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

class Measurement:
    def __init__(self, crop_predict, weed_predict, crop_label, weed_label, shape, total_classes):
        self.crop_predict = crop_predict
        self.weed_predict = weed_predict
        self.crop_label = crop_label
        self.weed_label = weed_label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):

        self.crop_predict = np.reshape(self.crop_predict, self.shape)
        self.weed_predict = np.reshape(self.weed_predict, self.shape)
        self.crop_label = np.reshape(self.crop_label, self.shape)
        self.weed_label = np.reshape(self.weed_label, self.shape)

        crop_predict_count = np.bincount(self.crop_predict, minlength=self.total_classes)
        weed_predict_count = np.bincount(self.weed_predict, minlength=self.total_classes)
        crop_label_count = np.bincount(self.crop_label, minlength=self.total_classes)
        weed_label_count = np.bincount(self.weed_label, minlength=self.total_classes)
         
        crop_temp = self.total_classes * np.array(self.crop_label, dtype="int") + np.array(self.crop_predict, dtype="int")  # Get category metrics
        weed_temp = self.total_classes * np.array(self.weed_label, dtype="int") + np.array(self.weed_predict, dtype="int")  # Get category metrics
    
        crop_temp_count = np.bincount(crop_temp, minlength=self.total_classes * self.total_classes)
        weed_temp_count = np.bincount(weed_temp, minlength=self.total_classes * self.total_classes)

        crop_cm = np.reshape(crop_temp_count, [self.total_classes, self.total_classes])
        crop_cm = np.diag(crop_cm)
        weed_cm = np.reshape(weed_temp_count, [self.total_classes, self.total_classes])
        weed_cm = np.diag(weed_cm)

        crop_U = crop_label_count + crop_predict_count - crop_cm
        crop_U = np.delete(crop_U, 0)
        crop_cm = np.delete(crop_cm, 0)
        crop_iou = np.divide(crop_cm, crop_U + 1e-7)
        weed_U = weed_label_count + weed_predict_count - weed_cm
        weed_U = np.delete(weed_U, 0)
        weed_cm = np.delete(weed_cm, 0)
        weed_iou = np.divide(weed_cm, weed_U + 1e-7)

        miou = np.zeros([2], dtype=np.float32)
        miou[0] = crop_iou
        miou[1] = weed_iou
        miou = np.nanmean(miou)
        

        if weed_iou == float('NaN'):
            weed_iou = 0.
        if crop_iou == float('NaN'):
            crop_iou = 0.

        # accuracy = np.divide(TP_TN, TP_TN_FP_FN + 1e-7)
        return miou, crop_iou, weed_iou

    def F1_score_and_recall(self):  # recall - sensitivity

        self.crop_predict = np.reshape(self.crop_predict, self.shape)
        self.weed_predict = np.reshape(self.weed_predict, self.shape)
        self.crop_label = np.reshape(self.crop_label, self.shape)
        self.weed_label = np.reshape(self.weed_label, self.shape)

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        self.axis1 = np.where(self.label != 2)
        self.predict1 = np.take(self.predict, self.axis1)
        self.label1 = np.take(self.label, self.axis1)

        TP_func1 = lambda predict: predict[:] == 1
        TP_func2 = lambda label,predict: label[:] == predict[:]
        TP = np.where(TP_func1(self.predict1) & TP_func2(self.label1, self.predict1), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        TN_func1 = lambda predict: predict[:] == 0
        TN_func2 = lambda label,predict: label[:] == predict[:]
        TN = np.where(TN_func1(self.predict1) & TN_func2(self.label1, self.predict1), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func1 = lambda predict: predict[:] == 0
        FN_func2 = lambda label,predict: label[:] != predict[:]
        FN = np.where((FN_func1(self.predict1) & FN_func2(self.label1,self.predict1)), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        FP_func1 = lambda predict: predict[:] == 1
        FP_func2 = lambda label,predict: label[:] != predict[:]
        FP = np.where((FP_func1(self.predict1) & FP_func2(self.label1,self.predict1)), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        self.axis2 = np.where(self.label == 2)
        self.predict2 = np.take(self.predict, self.axis2)
        self.label2 = np.take(self.label, self.axis2)

        FN_func3 = lambda predict: predict[:] == 2
        FN_func4 = lambda label,predict: label[:] != predict[:]
        FN2 = np.where((FN_func3(self.predict2) & FN_func4(self.label2,self.predict2)), 1, 0)
        FN = np.sum(FN2, dtype=np.int32) + FN

        FP_func3 = (lambda predict: predict[:] == 1)
        FP_func4 = (lambda predict: predict[:] == 0)
        FP_func5 = (FP_func3(self.predict2) | FP_func4(self.predict2))
        FP_func6 = lambda label,predict: label[:] != predict[:]
        FP2 = np.where(FP_func5 & FP_func6(self.label2,self.predict2), 1, 0)
        FP = np.sum(FP2, dtype=np.int32) + FP


        TP_FP = (TP + FP) + 1e-7

        TP_FN = (TP + FN) + 1e-7

        out = np.zeros((1))
        Precision = np.divide(TP, TP_FP)
        Recall = np.divide(TP, TP_FN)

        Pre_Re = (Precision + Recall) + 1e-7

        F1_score = np.divide(2. * (Precision * Recall), Pre_Re)

        accuracy = np.divide(TP + TN, TP + TN + FP + FN + 1e-7)

        return F1_score, Recall, accuracy

    def TDR(self): # True detection rate

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        self.axis1 = np.where(self.label != 2)
        self.predict1 = np.take(self.predict, self.axis1)
        self.label1 = np.take(self.label, self.axis1)

        TP_func1 = lambda predict: predict[:] == 1
        TP_func2 = lambda label,predict: label[:] == predict[:]
        TP = np.where(TP_func1(self.predict1) & TP_func2(self.label1, self.predict1), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        TN_func1 = lambda predict: predict[:] == 0
        TN_func2 = lambda label,predict: label[:] == predict[:]
        TN = np.where(TN_func1(self.predict1) & TN_func2(self.label1, self.predict1), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func1 = lambda predict: predict[:] == 0
        FN_func2 = lambda label,predict: label[:] != predict[:]
        FN = np.where((FN_func1(self.predict1) & FN_func2(self.label1,self.predict1)), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        FP_func1 = lambda predict: predict[:] == 1
        FP_func2 = lambda label,predict: label[:] != predict[:]
        FP = np.where((FP_func1(self.predict1) & FP_func2(self.label1,self.predict1)), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        self.axis2 = np.where(self.label == 2)
        self.predict2 = np.take(self.predict, self.axis2)
        self.label2 = np.take(self.label, self.axis2)

        FN_func3 = lambda predict: predict[:] == 2
        FN_func4 = lambda label,predict: label[:] != predict[:]
        FN2 = np.where((FN_func3(self.predict2) & FN_func4(self.label2,self.predict2)), 1, 0)
        FN = np.sum(FN2, dtype=np.int32) + FN

        FP_func3 = (lambda predict: predict[:] == 1)
        FP_func4 = (lambda predict: predict[:] == 0)
        FP_func5 = (FP_func3(self.predict2) | FP_func4(self.predict2))
        FP_func6 = lambda label,predict: label[:] != predict[:]
        FP2 = np.where(FP_func5 & FP_func6(self.label2,self.predict2), 1, 0)
        FP = np.sum(FP2, dtype=np.int32) + FP

        TP_FP = (TP + FP) + 1e-7

        out = np.zeros((1))
        TDR = np.divide(FP, TP_FP)

        TDR = 1 - TDR

        return TDR

#import matplotlib.pyplot as plt

#if __name__ == "__main__":

    
#    path = os.listdir("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels")

#    b_buf = []
#    for i in range(len(path)):
#        img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/"+ path[i])
#        img = tf.image.decode_png(img, 1)
#        img = tf.image.resize(img, [513, 513], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#        img = tf.image.convert_image_dtype(img, tf.uint8)
#        img = tf.squeeze(img, -1)
#        #plt.imshow(img, cmap="gray")
#        #plt.show()
#        img = img.numpy()
#        a = np.reshape(img, [513*513, ])
#        print(np.max(a))
#        img = np.array(img, dtype=np.int32) # void클래스가 정말 12 인지 확인해봐야함
#        #img = np.where(img == 0, 255, img)

#        b = np.bincount(np.reshape(img, [img.shape[0]*img.shape[1],]))
#        b_buf.append(len(b))
#        total_classes = len(b)  # 현재 124가 가장 많은 클래스수

#        #miou = MIOU(predict=img, label=img, total_classes=total_classes, shape=[img.shape[0]*img.shape[1],])
#        miou_ = Measurement(predict=img,
#                            label=img, 
#                            shape=[513*513, ], 
#                            total_classes=12).MIOU()
#        print(miou_)

#    print(np.max(b_buf))
