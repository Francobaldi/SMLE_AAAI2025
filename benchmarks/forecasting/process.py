import pandas as pd
import numpy as np
from core.data import *
from core.property import *
from core.model import *
from core.generate import *


name = 'forecasting'
data_dir = f'benchmarks/{name}/data'
res_dir = f'benchmarks/{name}/results'
prep_dir = f'benchmarks/{name}/preprocess'


#################################################################################################################
# Hyperpoarameters
#################################################################################################################
optimizer = 'adam'
loss = 'mse'
batch_size = 32
epochs = 1000
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
validation_split = 0.2
wind_in, wind_out = 8, 4
split_perc = 0.8
quantiles = [100, 95, 90]


#################################################################################################################
# Architectures
#################################################################################################################
layer_types = ['relu', 'lstm']
depths = [3,5]
boxes = ['constant', 'linear']
h_architectures = {
        (depth, layer_type) : [(width*wind_in*wind_out, layer_type) for width in range(1, int(depth/2)+1)] + \
                [(width*wind_in*wind_out, 'relu') for width in range(int(depth/2)+1, 0, -1)] \
        for depth in depths for layer_type in layer_types
        }
h_aux_architectures = {
        'constant' : wind_in*wind_out,
        'linear' : [(wind_in*wind_out, 'linear')]
        }


#################################################################################################################
# Data
#################################################################################################################
rrd = RegRealistic_Data(inst_by_period=20)
#rrd.get_data(data_dir)
rrd.data = pd.read_csv('benchmarks/forecasting/data/series.csv')
selected_series = pd.read_csv(f'{prep_dir}/selected.csv')
ids = selected_series['id'].unique()


#################################################################################################################
# Process
#################################################################################################################
for ID in ids:
    #################################################################################################################
    # Select Series -- Split -- Compute Windows
    #################################################################################################################
    series = rrd.get_series(ID=ID)
    series, series_train, series_test = rrd.split(series=series, split_perc=split_perc, std=True)
    x_train, y_train = rrd.expand_xy(series=series_train, wind_in=wind_in, wind_out=wind_out)
    x_test, y_test = rrd.expand_xy(series=series_test, wind_in=wind_in, wind_out=wind_out)

    x_train_delta, y_train_delta = rrd.apply_differencing(x_train, y_train)
    x_test_delta, y_test_delta = rrd.apply_differencing(x_test, y_test)


    #################################################################################################################
    # Property-Independent Unsafe Model (to initialize Postprocess)
    #################################################################################################################
    model_generator_delta = ModelGenerator(x_train=x_train_delta, y_train=y_train_delta, x_test=x_test_delta, y_test=y_test_delta,
                                           batch_size=batch_size, epochs=epochs,
                                           optimizer=optimizer, loss=loss, 
                                           validation_split=validation_split, callbacks=[early_stopping],
                                           mode='regression')
    model_generator = ModelGenerator(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                     batch_size=batch_size, epochs=epochs,
                                     optimizer=optimizer, loss=loss, 
                                     validation_split=validation_split, callbacks=[early_stopping],
                                     mode='regression')


    quantile, layer_type, depth, box = 'none', 'relu', 5, 'none'
    log_file = f'{res_dir}/unsafe_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
    h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
    unsafe = model_generator_delta.train(log_file=log_file, model_type='unsafe', h=h)
    model_generator_delta.reverse_differencing(log_file=log_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    log = model_generator.test(log_file=log_file)


    #################################################################################################################
    # Property
    #################################################################################################################
    for quantile in quantiles:
        sp = RegRealistic_Property(wind_in=wind_in, wind_out=wind_out)
        P, p, R, r = sp.generate(series, q=quantile)
        x_train_safe, y_train_safe = rrd.enforce_safety(x_train=x_train, y_train=y_train, P=P, p=p, R=R, r=r)

        x_train_safe_delta, y_train_safe_delta = rrd.apply_differencing(x_train_safe, y_train_safe)
        P_delta, p_delta, R_delta, r_delta = sp.apply_differencing(x_train_delta=x_train_safe_delta, x_test_delta=x_test_delta)

        model_generator_safe_delta = ModelGenerator(x_train=x_train_safe_delta, y_train=y_train_safe_delta, x_test=x_test_delta, y_test=y_test_delta,
                                               batch_size=batch_size, epochs=epochs,
                                               optimizer=optimizer, loss=loss, 
                                               validation_split=validation_split, callbacks=[early_stopping],
                                               mode='regression')
        model_generator_safe = ModelGenerator(x_train=x_train_safe, y_train=y_train_safe, x_test=x_test, y_test=y_test,
                                               batch_size=batch_size, epochs=epochs,
                                               optimizer=optimizer, loss=loss, 
                                               validation_split=validation_split, callbacks=[early_stopping],
                                               mode='regression')


        #################################################################################################################
        # Baseline
        #################################################################################################################
        layer_type, depth, box = 'relu', 5, 'none'
        print('======================================================================================================================')
        print(f'postprocess -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/postprocess_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not  os.path.isfile(log_file):
            postprocess = model_generator_delta.train(log_file=log_file, model_type='itp', P=P_delta, p=p_delta, R=R_delta, r=r_delta, unsafe=unsafe)
            model_generator_delta.reverse_differencing(log_file=log_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
        
        layer_type, depth, box = 'relu', 5, 'none'
        print('======================================================================================================================')
        print(f'preprocess -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/preprocess_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not  os.path.isfile(log_file):
            h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
            preprocess = model_generator_safe_delta.train(log_file=log_file, model_type='unsafe', P=P_delta, p=p_delta, R=R_delta, r=r_delta, h=h)
            model_generator_safe_delta.reverse_differencing(log_file=log_file, x_train=x_train_safe, y_train=y_train_safe, x_test=x_test, y_test=y_test)
            log = model_generator_safe.test(log_file=log_file, P=P, p=p, R=R, r=r)

        layer_type, depth, box = 'none', 'none', 'none'
        print('======================================================================================================================')
        print(f'oracle -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/oracle_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not os.path.isfile(log_file):
            oracle = model_generator.train(log_file=log_file, model_type='oracle', P=P, p=p, R=R, r=r)
            log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)

        
        #################################################################################################################
        # NoDiffSMLE
        #################################################################################################################
        layer_type, depth, box = 'relu', 5, 'constant'
        print('======================================================================================================================')
        print(f'nodiffsmle -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file_unsafe = f'{res_dir}/nodiffsmleunsafe_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        log_file = f'{res_dir}/nodiffsmle_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not os.path.isfile(log_file):
            try:
                h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
                h_lower = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_lower', output_init=-1.)
                h_upper = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_upper', output_init=1.)
                nodiffsmleunsafe = model_generator_safe.train(log_file=log_file_unsafe, model_type='smle', 
                                                              P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper, safe_train=False)
                log = model_generator_safe.test(log_file=log_file_unsafe, P=P, p=p, R=R, r=r)
                nodiffsmle = model_generator_safe.train(log_file=log_file, model_type='smle', 
                                                        P=P, p=p, R=R, r=r, 
                                                        h=nodiffsmleunsafe.h, h_lower=nodiffsmleunsafe.h_lower, h_upper=nodiffsmleunsafe.h_upper, 
                                                        g=nodiffsmleunsafe.g, g_poly=nodiffsmleunsafe.g_poly)
                log = model_generator_safe.test(log_file=log_file, P=P, p=p, R=R, r=r)
            except:
                pass


        #################################################################################################################
        # SMLE
        #################################################################################################################
        configs = [
                ('relu', 5, 'constant'), 
                ('lstm', 5, 'constant'),
                ('relu', 3, 'constant'), 
                ('relu', 5, 'linear')
                ]

        for layer_type, depth, box in configs:
            print('======================================================================================================================')
            print(f'smle -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
            print('======================================================================================================================')
            log_file_unsafe = f'{res_dir}/smleunsafe_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
            log_file = f'{res_dir}/smle_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
            if not os.path.isfile(log_file):
                try:
                    h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
                    h_lower = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_lower', output_init=-1.)
                    h_upper = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_upper', output_init=1.)
                    smleunsafe = model_generator_safe_delta.train(log_file=log_file_unsafe, model_type='smle', 
                                                                  P=P_delta, p=p_delta, R=R_delta, r=r_delta, h=h, h_lower=h_lower, h_upper=h_upper, safe_train=False)
                    model_generator_safe_delta.reverse_differencing(log_file=log_file_unsafe, x_train=x_train_safe, y_train=y_train_safe, x_test=x_test, y_test=y_test)
                    log = model_generator_safe.test(log_file=log_file_unsafe, P=P, p=p, R=R, r=r)
                    smle = model_generator_safe_delta.train(log_file=log_file, model_type='smle', 
                                                            P=P_delta, p=p_delta, R=R_delta, r=r_delta, 
                                                            h=smleunsafe.h, h_lower=smleunsafe.h_lower, h_upper=smleunsafe.h_upper, 
                                                            g=smleunsafe.g, g_poly=smleunsafe.g_poly)
                    model_generator_safe_delta.reverse_differencing(log_file=log_file, x_train=x_train_safe, y_train=y_train_safe, x_test=x_test, y_test=y_test)
                    log = model_generator_safe.test(log_file=log_file, P=P, p=p, R=R, r=r)
                except:
                    pass
