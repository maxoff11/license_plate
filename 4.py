from keras.models import load_model

loaded_model = load_model('symbol_recognition_model1.h5')
from keras.utils import plot_model

# Визуализация модели
plot_model(loaded_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)