def populate_resnet50_model(model):
    
    # Set model architecture details
    model.model_arch.set_details(
        arch = [
            ("ResNet50_preprocessing", "PreProcessingLayer", "(None, 224, 224, 3)", "0"),
            ("resnet50", "Functional", "(None, 2048)", "23,587,712"),
            ("dense_4", "Dense", "(None, 256)", "524,544"),
            ("batch_normalization_2", "BatchNormalization", "(None, 256)", "1,024"),
            ("activation_2", "Activation", "(None, 256)", "0"),
            ("dropout_2", "Dropout", "(None, 256)", "0"),
            ("dense_5", "Dense", "(None, 9)", "2,313"),
        ],
        params = [24_115_593, 527_369, 23_588_224],
        color = "green",
        icon = "check_circle",
        img_path = "assets/loss_res50.png",
        stop = 82
    )


    # ------------------ TRAIN METRICS ------------------
    train = model.train_metrics
    train.set_class_metrics("Cardboard",         0.9932, 0.9983, 0.9958, 588, 0.999976)
    train.set_class_metrics("Food Organics",     0.9943, 1.0000, 0.9971, 524, 0.999999)
    train.set_class_metrics("Glass",             0.9963, 0.9944, 0.9953, 536, 0.999985)
    train.set_class_metrics("Metal",             0.9921, 0.9941, 0.9931, 1010, 0.999974)
    train.set_class_metrics("Misc Trash",        0.9736, 0.9858, 0.9897, 632, 0.999946)
    train.set_class_metrics("Paper",             0.9969, 0.9738, 0.9953, 640, 0.999984)
    train.set_class_metrics("Plastic",           0.9898, 0.9889, 0.9894, 1176, 0.999918)
    train.set_class_metrics("Textile",           0.9902, 0.9975, 0.9739, 406, 0.999973)
    train.set_class_metrics("Vegetation",        0.9982, 0.9946, 0.9964, 556, 0.999998)
    
    train.set_macro_metrics(0.9938, 0.9942, 0.9940, auc=0.9999747438531593)
    train.set_weighted_metrics(0.9934, 0.9934, 0.9934, auc=0.9999680712837988)
    train.set_accuracy(0.9934)
    train.set_log_scale(1.9999)

    # ------------------ VAL METRICS ------------------
    val = model.val_metrics
    val.set_class_metrics("Cardboard",           0.8961, 0.9324, 0.9139, 74, 0.996847)
    val.set_class_metrics("Food Organics",       0.9242, 0.9242, 0.9242, 66, 0.997026)
    val.set_class_metrics("Glass",               0.8493, 0.9118, 0.8794, 68, 0.998035)
    val.set_class_metrics("Metal",               0.9113, 0.8898, 0.9004, 127, 0.991891)
    val.set_class_metrics("Misc Trash",          0.9155, 0.8125, 0.8609, 80, 0.978472)
    val.set_class_metrics("Paper",               0.9268, 0.9500, 0.9383, 80, 0.997259)
    val.set_class_metrics("Plastic",             0.8732, 0.8378, 0.8552, 148, 0.986508)
    val.set_class_metrics("Textile",             0.9057, 0.9412, 0.9231, 51, 0.998460)
    val.set_class_metrics("Vegetation",          0.9079, 0.9857, 0.9452, 70, 0.996068)
    
    val.set_macro_metrics(0.9011, 0.9095, 0.9045, auc=0.9933963141748706)
    val.set_weighted_metrics(0.8994, 0.8992, 0.8986, auc=0.992297005710006)
    val.set_accuracy(0.8992)
    val.set_log_scale(1.975)

    # ------------------ TEST METRICS ------------------
    test = model.test_metrics
    test.set_class_metrics("Cardboard",          0.9326, 0.8925, 0.9121, 93, 0.995079)
    test.set_class_metrics("Food Organics",      0.8795, 0.8795, 0.8795, 83, 0.995214)
    test.set_class_metrics("Glass",              0.9091, 0.9524, 0.9302, 84, 0.998317)
    test.set_class_metrics("Metal",              0.8916, 0.9367, 0.9136, 158, 0.993393)
    test.set_class_metrics("Misc Trash",         0.8817, 0.8283, 0.8542, 99, 0.983295)
    test.set_class_metrics("Paper",              0.9388, 0.9200, 0.9293, 100, 0.994461)
    test.set_class_metrics("Plastic",            0.9016, 0.8919, 0.8967, 185, 0.988191)
    test.set_class_metrics("Textile",            0.8788, 0.9062, 0.8923, 64, 0.996173)
    test.set_class_metrics("Vegetation",         0.9545, 0.9545, 0.9545, 88, 0.991431)
    
    test.set_macro_metrics(0.9076, 0.9069, 0.9069, auc=0.9928393740510233)
    test.set_weighted_metrics(0.9069, 0.9067, 0.9065, auc=0.9922101600997926)
    test.set_accuracy(0.9067)
    test.set_log_scale(1.98)
