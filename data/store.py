import streamlit as st

from data.b0 import populate_b0_model
from data.resnet50 import populate_resnet50_model
from data.resnet101 import populate_resnet101_model
from data.vgg16 import populate_vgg16_model

# Model Class for Model Architecture
class ModelArch:
    def __init__(self):
        self.arch = []
        self.num_params = []
        self.loss_img_path = ""
        self.epoch_stop = 0
        self.color = ""
        self.icon = ""
    
    def set_details(self, arch, params, color, icon, img_path, stop):
        self.arch = arch
        self.num_params = params
        self.color = color
        self.icon = icon
        self.loss_img_path = img_path
        self.epoch_stop = stop
    
    def get_params(self, idx):
        return f"{self.num_params[idx]:,}"

    def get_icon(self):
        return f":{self.color}[:material/{self.icon}:]"

# Model Class for per-class Metrics
class ClassMetrics:
    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.support = 0.0
        self.auc = 0.0

    def set_metrics(self, precision, recall, f1_score, support, auc):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.support = support
        self.auc = auc
    
    def get_metrics(self, field_name):
        value = getattr(self, field_name, None)
        if value is None:
            raise AttributeError(f"'{field_name}' is not a valid field.")
        return f"{value*100:.4g}%"


# Model Class for Total Performance Metrics
class Metrics:
    def __init__(self):
        self.classes = ["Cardboard", "Food Organics", "Glass", "Metal", "Misc Trash", "Paper", "Plastic", "Textile", "Vegetation"]
        self.class_metrics = {class_: ClassMetrics() for class_ in self.classes}
        self.macro_metrics = ClassMetrics()
        self.weighted_metrics = ClassMetrics()
        self.accuracy = 0.0
        self.log_scale = 1.99

    # Setters
    def set_class_metrics(self, class_name, precision, recall, f1_score, support, auc):
        if class_name in self.class_metrics:
            self.class_metrics[class_name].set_metrics(precision, recall, f1_score, support, auc)

    def set_macro_metrics(self, precision, recall, f1_score, auc):
        self.macro_metrics.set_metrics(precision, recall, f1_score, self.get_support(), auc)

    def set_weighted_metrics(self, precision, recall, f1_score, auc):
        self.weighted_metrics.set_metrics(precision, recall, f1_score, self.get_support(), auc)

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy
    
    def set_log_scale(self, lr):
        self.log_scale = lr

    # Getters
    def get_support(self):
        return sum([self.class_metrics[key].support for key in self.class_metrics.keys()])
    
    def get_accuracy(self):
        return f"{self.accuracy*100:.2f}%"

    def get_class_report(self):
        pr, re, f1, su = [], [], [], []
        for key in self.class_metrics:
            pr.append(self.class_metrics[key].precision)
            re.append(self.class_metrics[key].recall)
            f1.append(self.class_metrics[key].f1_score)
            su.append(self.class_metrics[key].support)
        
        return {
            "Class": self.classes,
            "Precision": pr,
            "Recall": re,
            "F1-Score": f1,
            "Support": su
        }

    def get_auc_scores(self):
        return {
            key: value.auc for key, value in self.class_metrics.items()
        }



# Model Class for Model Store
class ModelStore:
    def __init__(self, name):
        self.model_name = name
        self.model_arch = ModelArch()
        
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()
        self.test_metrics = Metrics()
        
        self.populate_data(name)

    def populate_data(self, name):
        if name == "EfficientNetB0":
            populate_b0_model(self)
        
        elif name == "ResNet50":
            populate_resnet50_model(self)
        
        elif name == "ResNet101":
            populate_resnet101_model(self)
        
        elif name == "VGG16":
            populate_vgg16_model(self)

        else:
            st.write("Invalid Model")
    
    def get_overall_metrics(self):
        return {
            "Accuracy": f"{self.test_metrics.accuracy*100:.4g}%",
            "Precision": f"{self.test_metrics.weighted_metrics.precision*100:.4g}%",
            "Recall": f"{self.test_metrics.weighted_metrics.recall*100:.4g}%",
            "F1-Score": f"{self.test_metrics.weighted_metrics.f1_score*100:.4g}%",
            "AUC": f"{self.test_metrics.weighted_metrics.auc*100:.4g}%",
        }.items()