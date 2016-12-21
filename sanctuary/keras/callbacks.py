from keras.callbacks import Callback
import numpy as np

from sanctuary.keras.capture import CaptureStdout

# source: https://github.com/IDSIA/sacred/issues/110
class TrainingHistoryToSacredInfo(Callback):
    """
    Stores the Keras training history to the Sacred info so that it can be
    observed during the training process.

    Usage example:

    @ex.capture
    def train(_run):
        # ...
        model.fit(X_train, y_train, callbacks=[TrainingHistoryToSacredInfo(_run)])
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

class ModelSummaryToSacredInfo(Callback):
    """
    Stores the model summary string and total number of parameters into sacred
    run info keys 'model_summary' and 'total_params'. It's invoked before
    training begins.

    Usage example:

    @ex.capture
    def train(_run):
        # ...
        model.fit(X_train, y_train, callbacks=[
            ModelSummaryToSacredInfo(_run)
        ])
    """
    def __init__(self, model, run):
        super(ModelSummaryToSacredInfo, self).__init__()
        self.model = model
        self.run = run

    def on_train_begin(self, logs={}):
        self.run.info['model_summary'] = model_summary(self.model)
        self.run.info['total_params'] = self.model.count_params()

def model_summary(model):
    """
    Returns the Keras model summary as a string instead of printing it to the
    stdout.
    """
    with CaptureStdout() as output:
        model.summary()
    return str(output)

class EvaluateOnSplit(Callback):
    """
    Evaluates the metrics on a split (eg. training) in test mode at the end of
    the epoch (so that's more comparable to evaluation on validation split).
    This is different than the evaluation after each mini-batch in training
    mode.

    It modifies the logs input parameter. The History callback is the last one,
    so it picks up these values. Also the CSVLogger should be placed after this
    one.

    Usage example:

    model.fit(X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size, nb_epoch=epochs,
        callbacks=[
            EvaluateOnSplit(model, X_train, y_train, batch_size=batch_size)
        ])
    """
    def __init__(self, model, X, y, split='train', **kwargs):
        super(EvaluateOnSplit, self).__init__()
        self.model = model
        self.X = X
        self.y = y
        self.split = split
        self.kwargs = kwargs

    def on_epoch_end(self, epoch, logs={}):
        names = self.model.metrics_names
        values = self.model.evaluate(self.X, self.y, **self.kwargs)
        for name, value in zip(names, np.atleast_1d(values)):
            logs[self.split + '_' + name] = value
