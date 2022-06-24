from PyQt5 import QtWidgets, uic
# from PyQt5 import QtGui
import sys
import os
import pickle
import argparse
import string

from DSIWasvaniQuestions import NQ_IR as NQ_IR_Wasvani
from DSICORD19 import NQ_IR as NQ_IR_Cord19
from DSIMMARCO import NQ_IR as NQ_IR_MMarco


# To use the latest checkpoint, set this variable to True.
# For training, set this to False.
RESUME_CHECKPOINT = True
TASK_TYPE = "indexing_retrieval"  # or "indexing_retrieval"

# Ratio is the number of queries present in the training dataset
QUERY_INSTANCE_RATIO_IN_TRAINING_DATA = 0.8

df_waswani_document = pickle.load(open("wasvanidataframe.pkl", "rb"))
df_waswani_document['doc_id'] = df_waswani_document['doc_id'].astype('str')
df_waswani_document_copy = df_waswani_document.copy()
df_waswani_query = pickle.load(open("wasvanidataframe(query).pkl", "rb"))
df_waswani_query['doc_id'] = df_waswani_query['doc_id'].astype('str')
data_len_waswani = len(df_waswani_document)


df_mmarco_document = pickle.load(open("mmarcodataframe.pkl", "rb"))
df_mmarco_document['doc_id'] = df_mmarco_document['doc_id'].astype('str')
df_mmarco_document_copy = df_mmarco_document.copy()
df_mmarco_query = pickle.load(open("mmarcodataframe(query).pkl", "rb"))
df_mmarco_query['doc_id'] = df_mmarco_query['doc_id'].astype('str')
data_len_mmarco = len(df_mmarco_document)

df_cord19_document = pickle.load(open("cord19dataframe.pkl", "rb"))
df_cord19_document['doc_id'] = df_cord19_document['doc_id'].astype('str')
df_cord19_document_copy = df_cord19_document.copy()
df_cord19_query = pickle.load(open("cord19dataframe(query).pkl", "rb"))
df_cord19_query['doc_id'] = df_cord19_query['doc_id'].astype('str')
data_len_cord19 = len(df_cord19_document)

data_len = {}
data_len['waswani'] = data_len_waswani
data_len['mmarco'] = data_len_mmarco
data_len['cord19'] = data_len_cord19

model_name = "t5-small"
token_len = 512  # deep tokenizer's output size currently 512
model_prefix = f"{model_name}-{token_len}"


def load_model(class_name="waswani", NQ_IR_Class=NQ_IR_Wasvani):

    # checkpoints_dir = 't5-small-512_wasvani_rows_checkpoint/'
    checkpoints_dir = \
        f"""{class_name}_{
            QUERY_INSTANCE_RATIO_IN_TRAINING_DATA
            }_{model_prefix}_{str(data_len[class_name])}_rows_checkpoint/"""
    checkpoint_files = sorted(os.listdir(checkpoints_dir))
    resume_from_checkpoint_path = checkpoints_dir

    if len(checkpoint_files) == 0:
        resume_from_checkpoint_path = None
        raise Exception('No checkpoint found')
    else:
        resume_from_checkpoint_path = (
            checkpoints_dir + checkpoint_files[-1])
    print(resume_from_checkpoint_path)

    args_dict = dict(
        # output_dir: path to save the checkpoints
        output_dir=f"./{model_prefix}_{str(data_len)}_rows_final",
        log_dir="logs",
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        max_input_length=token_len,
        max_output_length=token_len,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        # TODO: Change it to 64
        train_batch_size=64,
        eval_batch_size=1,
        num_train_epochs=50,
        gradient_accumulation_steps=4,
        # Number Of gpu
        n_gpu=1,
        resume_from_checkpoint_path=resume_from_checkpoint_path,
        val_check_interval=1,
        check_val_every_n_epoch=1,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        # fp_16: if you want to enable 16-bit training
        # then install apex and set this to true
        fp_16=False,
        # opt_level: you can find out more on optimisation levels here
        # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        # opt_level='O1',
        # max_grad_norm: if you enable 16-bit training then set this to
        # a sensible value, 0.5 is a good default
        max_grad_norm=1.0,
        seed=42,
    )

    args = argparse.Namespace(**args_dict)

    trained_model = NQ_IR_Class.load_from_checkpoint(
        resume_from_checkpoint_path, hparams=args)
    return trained_model


def predict(trained_model, query, tokenizer):
    def normalize_answer(s):

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix((remove_punc(lower(s))))

    def lmap(f, x):
        return list(map(f, x))

    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # print("gen_text: ", gen_text)
        return lmap(str.strip, gen_text)

    input_ = "retrieval:" + query
    if input_.startswith("retrieval"):
        max_length = 20
    else:
        max_length = 4096

    input_ = query.strip()

    source = tokenizer.batch_encode_plus(
        [input_], max_length=max_length,
        padding='max_length', truncation=True, return_tensors="pt")

    source_ids = source["input_ids"].squeeze()
    src_mask = source["attention_mask"].squeeze()

    generated_ids = trained_model.model.generate(
        source_ids,
        attention_mask=src_mask,
        use_cache=True,
        max_length=5,
        num_beams=20,
        num_return_sequences=20)

    preds = ids_to_clean_text(generated_ids)
    preds = [normalize_answer(s) for s in preds if len(s) > 0 and s != " "]
    return preds


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('GroupIR_Layout.ui', self)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.actionExit_2.triggered.connect(self.closeApplication)

        self.radioButton.setChecked(True)  # Wasvani is selected by default
        print(dir(self))

        # Loading models
        self.trained_models = []
        self.trained_models.append(
            load_model(class_name="waswani", NQ_IR_Class=NQ_IR_Wasvani))
        self.trained_models.append(
            load_model(class_name="mmarco", NQ_IR_Class=NQ_IR_MMarco))
        self.trained_models.append(
            load_model(class_name="cord19", NQ_IR_Class=NQ_IR_Cord19))

        # Create all models and save it in self
        # When button clicked, then according to the selected dataset,
        # execute the query on the selected model.

    def pushButtonClicked(self):
        query_text = self.plainTextEdit.toPlainText()
        selected_dataset = "None"

        if self.radioButton.isChecked():
            selected_dataset = "waswani"
            trained_model = self.trained_models[0]
        elif self.radioButton_2.isChecked():
            selected_dataset = "mmarco"
            trained_model = self.trained_models[1]
        elif self.radioButton_3.isChecked():
            selected_dataset = "cord19"
            trained_model = self.trained_models[2]
        # TODO:
        predict(trained_model, query_text, trained_model.tokenizer)

        print("Query text: "+query_text)
        print("Selected dataset: "+selected_dataset)
        print("Button clicked...")
        text_ = "1-deneme\n2-1234\n3-5678"
        self.label_2.setText(text_)

    def closeApplication(self):
        QtWidgets.QApplication.quit()


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
