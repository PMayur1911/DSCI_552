
def populate_b0_model(model):
    # Architecture details (placeholder, edit as needed)
    model.model_arch.set_details(
        arch = [
            ("EfficientNetB0_preprocessing", "PreProcessingLayer", "(None, 224, 224, 3)", "0"),
            ("efficientnetb0", "Functional", "(None, 1280)", "4,049,571"),
            ("dense_2", "Dense", "(None, 256)", "327,936"),
            ("batch_normalization_1", "BatchNormalization", "(None, 256)", "1,024"),
            ("activation_1", "Activation", "(None, 256)", "0"),
            ("dropout_1", "Dropout", "(None, 256)", "0"),
            ("dense_3", "Dense", "(None, 9)", "2,313"),
        ],
        params = [4_380_844, 330_761, 4_050_083],
        color = "blue",
        icon = "speed",
        img_path="assets/loss_b0.png",
        stop=71
    )
    
    ### TRAIN METRICS ###
    train = model.train_metrics
    train_metrics = [
        ("Cardboard", 0.9521, 0.9133, 0.9323, 588, 0.997429),
        ("Food Organics", 0.8142, 0.9866, 0.8921, 524, 0.998887),
        ("Glass", 0.9938, 0.5989, 0.7474, 536, 0.996088),
        ("Metal", 0.8736, 0.9584, 0.9141, 1010, 0.995472),
        ("Misc Trash", 0.8742, 0.9019, 0.8879, 632, 0.993552),
        ("Paper", 0.9556, 0.9422, 0.9489, 640, 0.998261),
        ("Plastic", 0.8775, 0.9201, 0.8983, 1176, 0.991127),
        ("Textile", 0.9550, 0.9409, 0.9479, 406, 0.998791),
        ("Vegetation", 0.9866, 0.9263, 0.9555, 556, 0.999533)
    ]
    for c in train_metrics:
        train.set_class_metrics(*c)
    train.set_macro_metrics(0.9203, 0.8987, 0.9027, auc=0.9965710738381348)
    train.set_weighted_metrics(0.9120, 0.9056, 0.9032, auc=0.9958572122659182)
    train.set_accuracy(0.9056)
    train.set_log_scale(1.99)

    ### VAL METRICS ###
    val = model.val_metrics
    val_metrics = [
        ("Cardboard", 0.7500, 0.8108, 0.7792, 74, 0.984371),
        ("Food Organics", 0.7857, 1.0000, 0.8800, 66, 0.996549),
        ("Glass", 0.9487, 0.5441, 0.6916, 68, 0.990027),
        ("Metal", 0.7066, 0.9291, 0.8027, 127, 0.983646),
        ("Misc Trash", 0.8657, 0.7250, 0.7891, 80, 0.972021),
        ("Paper", 0.7412, 0.7875, 0.7636, 80, 0.979861),
        ("Plastic", 0.8538, 0.7500, 0.7986, 148, 0.963430),
        ("Textile", 0.8431, 0.8431, 0.8431, 51, 0.991035),
        ("Vegetation", 0.9836, 0.8571, 0.9160, 70, 0.990531)
    ]
    for c in val_metrics:
        val.set_class_metrics(*c)
    val.set_macro_metrics(0.8309, 0.8052, 0.8071, auc=0.983496846136258)
    val.set_weighted_metrics(0.8225, 0.8063, 0.8040, auc=0.9809931943615443)
    val.set_accuracy(0.8063)
    val.set_log_scale(1.96)

    ### TEST METRICS ###
    test = model.test_metrics
    test_metrics = [
        ("Cardboard", 0.8333, 0.7527, 0.7910, 93, 0.980967),
        ("Food Organics", 0.7069, 0.9880, 0.8241, 83, 0.996390),
        ("Glass", 0.9773, 0.5119, 0.6719, 84, 0.989286),
        ("Metal", 0.7487, 0.9241, 0.8272, 158, 0.981943),
        ("Misc Trash", 0.7300, 0.7374, 0.7337, 99, 0.955213),
        ("Paper", 0.7523, 0.8200, 0.7847, 100, 0.975293),
        ("Plastic", 0.8333, 0.7568, 0.7932, 185, 0.961136),
        ("Textile", 0.7344, 0.7344, 0.7344, 64, 0.971541),
        ("Vegetation", 0.9595, 0.8068, 0.8765, 88, 0.990815)
    ]
    for c in test_metrics:
        test.set_class_metrics(*c)
    test.set_macro_metrics(0.8084, 0.7813, 0.7818, auc=0.9780648175419666)
    test.set_weighted_metrics(0.8068, 0.7904, 0.7873, auc=0.9763659654399166)
    test.set_accuracy(0.7904)
    test.set_log_scale(1.96)
