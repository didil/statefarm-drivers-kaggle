from keras.models import load_model
model = load_model('statefarm_drivers_4.h5')
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)