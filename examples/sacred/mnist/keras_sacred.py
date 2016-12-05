from keras.callbacks import Callback

# source: https://github.com/IDSIA/sacred/issues/110
class TrainingHistoryToSacredInfo(Callback):
    """
    Stores the Keras training history to the Sacred info so that it can be
    observed during the training process.

    Usage example:

    @ex.capture
    def train(_run):
        # ...
        model.fit(X_train, Y_train, callbacks=[TrainingHistoryToSacredInfo(_run)])
    """
    def __init__(self, run):
        super(TrainingHistoryToSacredInfo, self).__init__()
        self.output_key = 'training_history'
        self.run = run
        self.run.info[self.output_key] = {}

    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            log_out = self.run.info[self.output_key].get(k, [])
            log_out.append(v)
            self.run.info[self.output_key][k] = log_out
