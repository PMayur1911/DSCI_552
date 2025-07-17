def populate_vgg16_model(model):

    # Set model architecture details
    model.model_arch.set_details(
        arch = [
            ("VGG16_preprocessing", "PreProcessingLayer", "(None, 224, 224, 3)", "0"),
            ("vgg16", "Functional", "(None, 512)", "14,714,688"),
            ("dense_10", "Dense", "(None, 256)", "131,328"),
            ("batch_normalization_5", "BatchNormalization", "(None, 256)", "1,024"),
            ("activation_5", "Activation", "(None, 256)", "0"),
            ("dropout_5", "Dropout", "(None, 256)", "0"),
            ("dense_11", "Dense", "(None, 9)", "2,313"),
        ],
        params = [14_849_353, 134_153, 14_715_200],
        color = "red",
        icon = "bug_report",
        img_path="assets/loss_vgg16.png",
        stop=58
    )

    # ------------------ TRAIN METRICS ------------------
    train = model.train_metrics
    train.set_class_metrics("Cardboard",         0.9723, 0.9558, 0.9640, 588, 0.999026)
    train.set_class_metrics("Food Organics",     0.9731, 0.9656, 0.9693, 524, 0.999486)
    train.set_class_metrics("Glass",             0.9682, 0.9664, 0.9673, 536, 0.999221)
    train.set_class_metrics("Metal",             0.9540, 0.9644, 0.9591, 1010, 0.998588)
    train.set_class_metrics("Misc Trash",        0.9120, 0.9351, 0.9234, 632, 0.997297)
    train.set_class_metrics("Paper",             0.9406, 0.9656, 0.9530, 640, 0.999085)
    train.set_class_metrics("Plastic",           0.9657, 0.9337, 0.9494, 1176, 0.997797)
    train.set_class_metrics("Textile",           0.9724, 0.9557, 0.9640, 406, 0.999265)
    train.set_class_metrics("Vegetation",        0.9581, 0.9874, 0.9725, 556, 0.999755)

    train.set_macro_metrics(0.9574, 0.9589, 0.9580, auc=0.9988355817139583)
    train.set_weighted_metrics(0.9568, 0.9565, 0.9565, auc=0.998680860154345)
    train.set_accuracy(0.9565)
    train.set_log_scale(1.995)

    # ------------------ VAL METRICS ------------------
    val = model.val_metrics
    val.set_class_metrics("Cardboard",           0.8400, 0.8514, 0.8456, 74, 0.992695)
    val.set_class_metrics("Food Organics",       0.8841, 0.9242, 0.9037, 66, 0.997808)
    val.set_class_metrics("Glass",               0.8824, 0.8824, 0.8824, 68, 0.984301)
    val.set_class_metrics("Metal",               0.8261, 0.8976, 0.8604, 127, 0.983325)
    val.set_class_metrics("Misc Trash",          0.7683, 0.7875, 0.7778, 80, 0.978765)
    val.set_class_metrics("Paper",               0.8333, 0.8125, 0.8228, 80, 0.986641)
    val.set_class_metrics("Plastic",             0.8657, 0.7838, 0.8227, 148, 0.978008)
    val.set_class_metrics("Textile",             0.8113, 0.8431, 0.8269, 51, 0.990842)
    val.set_class_metrics("Vegetation",          0.9552, 0.9143, 0.9343, 70, 0.996336)

    val.set_macro_metrics(0.8518, 0.8552, 0.8530, auc=0.9876355565139249)
    val.set_weighted_metrics(0.8507, 0.8495, 0.8493, auc=0.9861040761751061)
    val.set_accuracy(0.8495)
    val.set_log_scale(1.96)

    # ------------------ TEST METRICS ------------------
    test = model.test_metrics
    test.set_class_metrics("Cardboard",          0.8837, 0.8172, 0.8492, 93, 0.989547)
    test.set_class_metrics("Food Organics",      0.8090, 0.8675, 0.8372, 83, 0.988256)
    test.set_class_metrics("Glass",              0.8111, 0.8690, 0.8391, 84, 0.993227)
    test.set_class_metrics("Metal",              0.8500, 0.8608, 0.8553, 158, 0.982309)
    test.set_class_metrics("Misc Trash",         0.8021, 0.7778, 0.7897, 99, 0.960364)
    test.set_class_metrics("Paper",              0.8529, 0.8700, 0.8614, 100, 0.989953)
    test.set_class_metrics("Plastic",            0.7833, 0.7622, 0.7726, 185, 0.970337)
    test.set_class_metrics("Textile",            0.7971, 0.8594, 0.8271, 64, 0.988746)
    test.set_class_metrics("Vegetation",         0.9512, 0.8864, 0.9176, 88, 0.990290)

    test.set_macro_metrics(0.8378, 0.8411, 0.8388, auc=0.9836698624992883)
    test.set_weighted_metrics(0.8345, 0.8333, 0.8334, auc=0.981863633453042)
    test.set_accuracy(0.8333)
    test.set_log_scale(1.96)
