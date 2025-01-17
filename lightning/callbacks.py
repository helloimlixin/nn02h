from lightning.pytorch.callbacks import EarlyStopping, Callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super(MyPrintingCallback, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done :)")
