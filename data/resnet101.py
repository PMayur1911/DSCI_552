def populate_resnet101_model(model):
    # Set model architecture details
    model.model_arch.set_details(
        arch = [
            ("ResNet101_preprocessing", "PreProcessingLayer", "(None, 224, 224, 3)", "0"),
            ("resnet101", "Functional", "(None, 2048)", "42,658,176"),
            ("dense_6", "Dense", "(None, 512)", "1,049,088"),
            ("batch_normalization_3", "BatchNormalization", "(None, 512)", "2,048"),
            ("activation_3", "Activation", "(None, 512)", "0"),
            ("dropout_3", "Dropout", "(None, 512)", "0"),
            ("dense_7", "Dense", "(None, 9)", "4,617"),
        ],
        params = [43_713_929, 1_054_729, 42_659_200],
        color = "orange",
        icon = "storage",
        img_path = "assets/loss_res101.png",
        stop = 59
    )

    # ------------------ TRAIN METRICS ------------------
    train = model.train_metrics
    train.set_class_metrics("Cardboard", 1.0000, 0.9983, 0.9991, 588, 0.999999)
    train.set_class_metrics("Food Organics", 0.9962, 1.0000, 0.9981, 524, 1.0)
    train.set_class_metrics("Glass", 0.9981, 1.0000, 0.9991, 536, 0.999996)
    train.set_class_metrics("Metal", 0.9990, 0.9960, 0.9975, 1010, 0.999994)
    train.set_class_metrics("Misc Trash", 0.9952, 0.9905, 0.9929, 632, 0.999935)
    train.set_class_metrics("Paper", 0.9953, 0.9984, 0.9969, 640, 0.999996)
    train.set_class_metrics("Plastic", 0.9949, 0.9949, 0.9949, 1176, 0.999973)
    train.set_class_metrics("Textile", 0.9902, 0.9975, 0.9939, 406, 0.999998)
    train.set_class_metrics("Vegetation", 1.0000, 0.9982, 0.9991, 556, 1.0)

    train.set_macro_metrics(0.9966, 0.9971, 0.9968, auc=0.9999877195825422)
    train.set_weighted_metrics(0.9967, 0.9967, 0.9967, auc=0.9999857987255197)
    train.set_accuracy(0.9967)
    train.set_log_scale(1.9999)

    # ------------------ VAL METRICS ------------------
    val = model.val_metrics
    val.set_class_metrics("Cardboard", 0.9189, 0.9189, 0.9189, 74, 0.995809)
    val.set_class_metrics("Food Organics", 0.9286, 0.9848, 0.9559, 66, 0.998068)
    val.set_class_metrics("Glass", 0.9130, 0.9265, 0.9197, 68, 0.996408)
    val.set_class_metrics("Metal", 0.8740, 0.8740, 0.8740, 127, 0.988702)
    val.set_class_metrics("Misc Trash", 0.8571, 0.8250, 0.8408, 80, 0.982639)
    val.set_class_metrics("Paper", 0.9167, 0.9625, 0.9390, 80, 0.998246)
    val.set_class_metrics("Plastic", 0.8993, 0.8446, 0.8711, 148, 0.985346)
    val.set_class_metrics("Textile", 0.8364, 0.9020, 0.8679, 51, 0.996590)
    val.set_class_metrics("Vegetation", 0.9855, 0.9714, 0.9784, 70, 0.991993)

    val.set_macro_metrics(0.9033, 0.9122, 0.9073, auc=0.992644411270289)
    val.set_weighted_metrics(0.9018, 0.9018, 0.9014, auc=0.9914276720703857)
    val.set_accuracy(0.9018)
    val.set_log_scale(1.98)

    # ------------------ TEST METRICS ------------------
    test = model.test_metrics
    test.set_class_metrics("Cardboard", 0.9111, 0.8817, 0.8962, 93, 0.994830)
    test.set_class_metrics("Food Organics", 0.8953, 0.9277, 0.9112, 83, 0.995214)
    test.set_class_metrics("Glass", 0.9259, 0.8929, 0.9091, 84, 0.996976)
    test.set_class_metrics("Metal", 0.8896, 0.9177, 0.9034, 158, 0.989528)
    test.set_class_metrics("Misc Trash", 0.7810, 0.8283, 0.8039, 99, 0.981948)
    test.set_class_metrics("Paper", 0.9192, 0.9100, 0.9146, 100, 0.996745)
    test.set_class_metrics("Plastic", 0.9249, 0.8649, 0.8939, 185, 0.988543)
    test.set_class_metrics("Textile", 0.8088, 0.8594, 0.8333, 64, 0.993715)
    test.set_class_metrics("Vegetation", 0.9326, 0.9432, 0.9379, 88, 0.992481)

    test.set_macro_metrics(0.8876, 0.8917, 0.8893, auc=0.9922199179336353)
    test.set_weighted_metrics(0.8926, 0.8910, 0.8914, auc=0.9915274279671469)
    test.set_accuracy(0.8910)
    test.set_log_scale(1.98)
