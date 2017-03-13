import os, time, ast, shutil
import json, geojson, geoio
import numpy as np
import subprocess
from net import VggNet
from mltools import geojson_tools as gt
from mltools import data_extractors as de
from gbdx_task_interface import GbdxTaskInterface


class TrainCnnClassifier(GbdxTaskInterface):

    def __init__(self):
        '''
        Instantiate string and data inputs, organize data for training
        '''
        GbdxTaskInterface.__init__(self)

        # Get string inputs
        self.start = time.time()
        self.classes = [i.strip() for i in self.get_input_string_port('classes').split(',')]
        self.two_rounds = ast.literal_eval(self.get_input_string_port('two_rounds', default='True'))
        self.filter_geojson = ast.literal_eval(self.get_input_string_port('filter_geojson', default='True'))
        self.min_side_dim = int(self.get_input_string_port('min_side_dim', default='10'))
        self.max_side_dim = int(self.get_input_string_port('max_side_dim', default='125'))
        self.train_size = int(self.get_input_string_port('train_size', default='10000'))
        self.batch_size = int(self.get_input_string_port('batch_size', default='32'))
        self.nb_epoch = int(self.get_input_string_port('nb_epoch', default='35'))
        self.use_lowest_val_loss = ast.literal_eval(self.get_input_string_port('use_lowest_val_loss', default='True'))
        self.nb_epoch_2 = int(self.get_input_string_port('nb_epoch_2', default='8'))
        self.train_size_2 = int(self.get_input_string_port('train_size_2', default=int(0.5 * self.train_size)))
        self.test = ast.literal_eval(self.get_input_string_port('test', default='True'))
        self.test_size = int(self.get_input_string_port('test_size', default='5000'))
        self.lr_1 = float(self.get_input_string_port('learning_rate', default='0.001'))
        self.lr_2 = float(self.get_input_string_port('learning_rate_2', default='0.01'))
        self.bit_depth = int(self.get_input_string_port('bit_depth', default='8'))
        self.kernel_size = int(self.get_input_string_port('kernel_size', default='3'))
        self.small_model = ast.literal_eval(self.get_input_string_port('small_model', default='False'))
        self.resize_dim = ast.literal_eval(self.get_input_string_port('resize_dim', default='None'))
        print '\nString args loaded. Running for {} seconds'.format(str(time.time() - self.start))

        # Get data input locations
        self.geoj_dir = self.get_input_data_port('geojson')
        self.img_dir = self.get_input_data_port('images')

        # Create necessary directories
        self.trained_model, self.model_weights = self._create_directories()
        geoj, self.bands = self._check_inputs()
        shutil.copyfile(geoj, os.path.join(self.img_dir, 'orig_geojson.geojson'))
        os.chdir(self.img_dir)


    def _check_inputs(self):
        '''
        Ensure proper composition of input directory ports and all images have same first
            dimension.
        Returns path to geojson file and number of bands used in imagery
        '''
        # Ensure proper number of images provided
        in_img_dir = os.listdir(self.img_dir)
        imgs = [img for img in in_img_dir if img.endswith('.tif')]
        if len(imgs) > 5:
            raise Exception('There are too many images in the input image directory. ' \
                            'Please use a maximum of five image strips.')
        if len(imgs) == 0:
            raise Exception('No images were found in the input directory. Please ' \
                            'provide at lease one GeoTif image.')

        # Ensure all images have same number of bands
        bands = [geoio.GeoImage(os.path.join(self.img_dir, img)).shape[0] for img in imgs]
        if not all(dim == bands[0] for dim in bands):
            raise Exception('Please make sure all images have the same number of bands')

        # Ensure only one geojson
        geoj_list = [geoj for geoj in os.listdir(self.geoj_dir) if geoj.endswith('.geojson')]
        if len(geoj_list) != 1:
            raise Exception('Make sure there is exactly one geojson in image_dest s3 ' \
                            'bucket')

        return os.path.join(self.geoj_dir, geoj_list[0]), bands[0]


    def _create_directories(self):
        '''
        Create output directory ports
        '''
        # Create models dir for training
        os.makedirs(os.path.join(self.img_dir, 'models'))

        # Create output directories
        trained_model = self.get_output_data_port('trained_model')
        model_weights = os.path.join(trained_model, 'model_weights')
        os.makedirs(trained_model)
        os.makedirs(model_weights)
        os.makedirs(os.path.join(model_weights, 'round_1'))
        if self.two_rounds:
            os.makedirs(os.path.join(model_weights, 'round_2'))

        return trained_model, model_weights


    def make_tiled_images(self):
        '''
        Make images tiled using gdal translate. much faster when generating chips during
            training.
        '''
        currdir = os.path.abspath('.')

        # Navigate to image directory, get image list
        os.chdir(self.img_dir)
        imgs = [img for img in os.listdir('.') if img.endswith('.tif')]

        for img in imgs:
            cmd = 'gdal_translate -co TILED=YES {} {}_t.tif'.format(img, img.strip('.tif'))
            subprocess.call(cmd, shell=True)
            shutil.move('{}_t.tif'.format(img.strip('.tif')), img)

        os.chdir(currdir)

    def prep_geojsons(self, geoj):
        '''
        Prep input geojson as follows: filter, split into train/test and create a
            file with balanced classes (depending on input params)
        '''
        # Filter geojson
        if self.filter_geojson:
            gt.filter_polygon_size(geoj, output_file = 'filtered_geojson.geojson',
                                   min_side_dim = self.min_side_dim,
                                   max_side_dim = self.max_side_dim)
            geoj = 'filtered_geojson.geojson'

        # Create train/test data geojsons
        if self.test:
            gt.create_train_test(geoj, output_file='filtered.geojson',
                                 test_size=self.test_size)
            geoj = 'train_filtered.geojson'

        # Create balanced datasets
        if self.two_rounds:
            gt.create_balanced_geojson(geoj, classes=self.classes,
                                       output_file='train_balanced.geojson')
            geoj = 'train_balanced.geojson'

        # Establish number of remaining train polygons
        with open(geoj) as inp_file:
            poly_ct = len(geojson.load(inp_file)['features'])

        if poly_ct < self.train_size:
            raise Exception('There are only {} polygons that can be used as training ' \
                            'data, cannot train the network on {} samples. Please ' \
                            'decrease train_size or provide more ' \
                            'polygons.'.format(str(poly_ct), str(self.train_size)))

        # Return name of input file
        return geoj


    def fit_model(self, model, inp_geojson, rnd, input_shape, **kwargs):
        '''
        Fit model
        '''
        # Get nb_epoch and train_size params
        if rnd == 2:
            nb_epoch, train_size = self.nb_epoch_2, self.train_size_2
        else:
            nb_epoch, train_size = self.nb_epoch, self.train_size

        try:
            hist = model.fit_from_geojson(inp_geojson, nb_epoch=nb_epoch,
                                          max_side_dim=self.max_side_dim,
                                          min_side_dim=self.min_side_dim,
                                          validation_split=0.1, return_history=True,
                                          chips_per_batch=1000, train_size=train_size,
                                          **kwargs)
        except (MemoryError):
            raise Exception('Model does not fit in memory. Plase try one or more of '\
                            'the following:\n' \
                            '- Use resize_dim to downsample chips. Input images should '\
                            'be no larger than 250px \n' \
                            '- Use a smaller batch_size \n' \
                            '- Set the small_model flag to True')

        # Save weights to ouptput dir
        save_path = os.path.join(self.model_weights, 'round_{}'.format(str(rnd)))
        for weights in os.listdir('models/'):
            shutil.copy('models/' + weights, save_path)
        return model, hist


    def test_net(self, model):
        '''
        Get accuracy metrics for test data, save to test_report.txt
        '''

        y_pred, y_true = [], []
        with open('test_filtered.geojson') as f:
            test_data = geojson.load(f)['features']

        for polygon_ix in xrange(0, self.test_size, 1000):
            x, y = de.get_data_from_polygon_list(test_data[polygon_ix: polygon_ix + 1000],
                                                 max_side_dim = self.max_side_dim,
                                                 classes=self.classes,
                                                 bit_depth=self.bit_depth,
                                                 resize_dim=self.resize_dim)

            y_pred += list(model.model.predict_classes(x))
            y_true += [int(clss[1]) for clss in y]

        test_size = len(y_true)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        wrong, right = np.argwhere(y_pred != y_true), np.argwhere(y_pred == y_true)
        fp = int(np.sum([y_pred[i] for i in wrong]))
        tp = int(np.sum([y_pred[i] for i in right]))
        fn, tn = int(len(wrong) - fp), int(len(right) - tp)

        # get accuracy metrics
        try:
            precision = float(tp) / (tp + fp)
        except (ZeroDivisionError):
            precision = 'N/A'
        try:
            recall = float(tp)/(tp + fn)
        except (ZeroDivisionError):
            recall = 'N/A'

        test_report = 'Test size: ' + str(test_size) + \
                      '\nFalse Positives: ' + str(fp) + \
                      '\nFalse Negatives: ' + str(fn) + \
                      '\nPrecision: ' + str(precision) + \
                      '\nRecall: ' + str(recall) + \
                      '\nAccuracy: ' + str(float(len(right))/test_size)
        print test_report

        # Record test results
        with open(os.path.join(self.trained_model, 'test_report.txt'), 'w') as tr:
            tr.write(test_report)


    def invoke(self):
        '''
        Execute task
        '''

        # Determine input shape
        if self.resize_dim:
            input_shape = self.resize_dim
        else:
            input_shape = [self.bands, self.max_side_dim, self.max_side_dim]

        # Prep geojson for training
        inp = self.prep_geojsons('orig_geojson.geojson')

        # Training round 1
        net = VggNet(classes=self.classes, batch_size=self.batch_size,
                    input_shape=input_shape, learning_rate=self.lr_1,
                    kernel_size=self.kernel_size, small_model=self.small_model)

        net, hist = self.fit_model(model=net, inp_geojson=inp, rnd=1, input_shape=input_shape)

        # Find lowest val_loss, load weights
        if self.use_lowest_val_loss:
            val_losses = [epoch['val_loss'][0] for epoch in hist]
            min_epoch = np.argmin(val_losses)
            min_loss = val_losses[min_epoch]
            min_weights = 'models/epoch' + str(min_epoch) + '_{0:.2f}.h5'.format(min_loss)
            net.model.load_weights(min_weights)

        # Training round 2
        if self.two_rounds:
            # Remove old models
            mod_list = os.listdir('models')
            for mod in mod_list:
                os.remove(os.path.join('models', mod))

            net, _ = self.fit_model(model=net, inp_geojson='train_filtered.geojson', rnd=2,
                                  input_shape=input_shape, retrain=True,
                                  learning_rate_2 = self.lr_2)

        # Test network
        if self.test:
            self.test_net(model=p)

        # Save model architecture and weights to output directory
        os.chdir(self.trained_model)
        json_str = net.model.to_json()
        net.model.save_weights('model_weights.h5')
        with open('model_architecture.json', 'w') as arch:
            json.dump(json_str, arch)


if __name__ == '__main__':
    with TrainCnnClassifier() as task:
        task.invoke()
