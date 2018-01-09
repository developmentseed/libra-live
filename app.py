# coding=utf-8

from datetime import datetime
import json
import math
from StringIO import StringIO
import os
import uuid

from cachetools.func import lru_cache, rr_cache
from celery import Celery, chain, chord, states
from flask import Flask, redirect, request, send_from_directory, jsonify, url_for
from flask_cors import CORS
import mercantile
from pyproj import Proj
from mercantile import Tile
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, transform_bounds
from werkzeug.wsgi import DispatcherMiddleware


APPLICATION_ROOT = os.environ.get('APPLICATION_ROOT', '')
IMAGERY_PATH = os.environ.get('IMAGERY_PATH', 'tiles')
MIN_ZOOM = int(os.environ.get('MIN_ZOOM', 0))
MAX_ZOOM = int(os.environ.get('MAX_ZOOM', 22))
SERVER_NAME = os.environ.get('SERVER_NAME', None)
TASK_TIMEOUT = int(os.environ.get('TASK_TIMEOUT', 60 * 15))
USE_X_SENDFILE = os.environ.get('USE_X_SENDFILE', False)

# strip trailing slash if necessary
if IMAGERY_PATH[-1] == '/':
    IMAGERY_PATH = IMAGERY_PATH[:-1]

app = Flask('posm-imagery-api')
CORS(app)
app.config['APPLICATION_ROOT'] = APPLICATION_ROOT
app.config['SERVER_NAME'] = SERVER_NAME
app.config['USE_X_SENDFILE'] = USE_X_SENDFILE


def intensity_range(image, range_values='image', clip_negative=False):
    if range_values == 'dtype':
        range_values = image.dtype.type

    if range_values == 'image':
        i_min = np.min(image)
        i_max = np.max(image)
    else:
        i_min, i_max = range_values
    return i_min, i_max


def rescale_intensity(image, in_range='image', out_range='dtype'):
    dtype = image.dtype.type

    if in_range is None:
        in_range = 'image'
        msg = "`in_range` should not be set to None. Use {!r} instead."
        print(msg.format(in_range))

    if out_range is None:
        out_range = 'dtype'
        msg = "`out_range` should not be set to None. Use {!r} instead."
        print(msg.format(out_range))

    imin, imax = intensity_range(image, in_range)
    omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))

    image = np.clip(image, imin, imax)

    image = (image - imin) / float(imax - imin)
    return dtype(image * (omax - omin) + omin)


def color_map_reader(path):
    """
    reads the colormap from a text file given as path.
    """

    max_value = 255
    mode = None

    try:
        i = 0
        colormap = {}
        with open(path) as cmap:
            lines = cmap.readlines()
            for line in lines:
                if not mode:
                    if 'mode = ' in line:
                        mode = float(line.replace('mode = ', ''))
                    else:
                        continue
                else:
                    str = line.split()
                    if str == []:  # when there are empty lines at the end of the file
                        break
                    colormap.update(
                        {
                            i: (int(round(float(str[0]) * max_value / mode)),
                                int(round(float(str[1]) * max_value / mode)),
                                int(round(float(str[2]) * max_value / mode)))
                        }
                    )
                    i += 1
    except IOError:
        pass

    return colormap


@lru_cache()
def get_source(path):
    return rasterio.open(path)


def render_tile(id, bands, res=500, bounds=None, point=None):
    image_res = 30
    data = []
    path = id[3:6]
    row = id[6:9]

    if point:
        point = map(float, point.split(','))
        bounds = [point[0] - 0.2, point[1] - 0.2, point[0] + 0.2, point[1] + 0.2]

    elif bounds:
        bounds = map(float, bounds.split(','))

    window = [[0, 0], [0, 0]]

    for i, band in enumerate(bands):
        image_uri = 'http://landsat-pds.s3.amazonaws.com/L8/%s/%s/%s/%s_B%s.TIF' % (path, row, id, id, band)
        src = get_source(image_uri)
        print('Band %s' % band)
        if i == 0:
            if bounds:

                proj_to = Proj(**src.read_crs())

                tile_ul_proj = proj_to(bounds[0], bounds[3])
                tile_lr_proj = proj_to(bounds[2], bounds[1])
                tif_ul_proj = (src.bounds[0], src.bounds[3])

                # y, x (rows, columns)
                top = int((tif_ul_proj[1] - tile_ul_proj[1]) / image_res )
                left = int((tile_ul_proj[0] - tif_ul_proj[0]) / image_res)
                bottom = int((tif_ul_proj[1] - tile_lr_proj[1]) / image_res)
                right = int((tile_lr_proj[0] - tif_ul_proj[0]) / image_res)

                window = [
                          [top, bottom],
                          [left, right]
                ]


            data = np.empty(shape=(3, res, res)).astype(src.profile['dtype'])

        image = np.empty(shape=(1, res, res)).astype(src.profile['dtype'])
        image = src.read(out=image, window=window)

        p_low, p_high = np.percentile(image[np.logical_and(image > 0, image < 65535)], (2, 98))
        image = rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 255))

        data[i] = image

    return data

def render_viirs_tile(bands, res=500, bounds=None, point=None):
    image_res = 0.0041666667
    data = []

    if point:
        point = map(float, point.split(','))
        bounds = [point[0] - 0.2, point[1] - 0.2, point[0] + 0.2, point[1] + 0.2]

    elif bounds:
        bounds = map(float, bounds.split(','))

    window = [[0, 0], [0, 0]]

    image_uri = 'https://s3.amazonaws.com/devseed-kes-deployment/nightlight/nl_201711_NAmerica.tif'
    src = get_source(image_uri)

    # y, x (rows, columns)
    top = (75 - point[0]) / image_res - ( res / 2 )
    left = (point[1] + 180) / image_res - ( res / 2 )
    bottom = (75 - point[0]) / image_res + ( res / 2 )
    right = (point[1] + 180) / image_res + ( res / 2 )

    window = [
          [top, bottom],
          [left, right]
    ]

    image = np.empty(shape=(1, res, res)).astype(src.profile['dtype'])
    image = src.read(out=image, window=window)

    p_low, p_high = np.percentile(image, (1, 99))
    image = rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 255))

    data = np.empty(shape=(3, res, res)).astype(src.profile['dtype'])
    data[0] = image
    data[1] = image
    data[2] = image

    return data


class InvalidTileRequest(Exception):
    status_code = 404

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@rr_cache()
def read_tile(id, product, res=500, bounds=None, point=None):

    products = {
        'ndvi': [4, 5],
        'water': [5, 4, 3],
        'default': [4, 3, 2]
    }

    bands = products.get(product, products['default'])

    data = render_tile(id, bands, res, bounds=bounds, point=point)

    if product == 'ndvi':
        nir = data[1].astype('float32')
        red = data[0].astype('float32')

        ndvi = np.nan_to_num(np.true_divide((nir - red), (nir + red)))

        colormap = color_map_reader('/app/colormap.txt')
        def drew(val, b):
            return colormap[int(val)][b]

        vdrew = np.vectorize(drew)
        ndvi = rescale_intensity(ndvi, out_range=(0, 255))
        for i in range(0, 3):
            data[i] = vdrew(ndvi, i)

    imgarr = np.ma.transpose(data, [1, 2, 0]).astype(np.byte)

    print('Generating the Image')
    out = StringIO()
    im = Image.fromarray(imgarr, 'RGB')
    im.save(out, 'png')

    return out.getvalue()

@rr_cache()
def read_viirs_tile(res=500, bounds=None, point=None):

    data = render_viirs_tile(res, bounds=bounds, point=point)

    imgarr = np.ma.transpose(data, [1, 2, 0]).astype(np.byte)

    print('Generating the Image')
    out = StringIO()
    im = Image.fromarray(imgarr, 'RGB')
    im.save(out, 'png')

    return out.getvalue()


@app.errorhandler(InvalidTileRequest)
def handle_invalid_tile_request(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(IOError)
def handle_ioerror(error):
    return '', 404


# @app.route('/image/<id>')
# def get_image(id):
#     product = request.args.get('product', 'default')
#     resolution = int(request.args.get('resolution', 1)) * 500
#     point = request.args.get('point', None)
#     bounds = request.args.get('bounds', None)
#     tile = read_tile(id, product, resolution, bounds=bounds, point=point)
#
#     return tile, 200, {
#         'Content-Type': 'image/png'
#     }

@app.route('/image/viirs')
def get_image():
    resolution = int(request.args.get('resolution', 1)) * 500
    point = request.args.get('point', None)
    bounds = request.args.get('bounds', None)
    tile = read_viirs_tile(resolution, bounds=bounds, point=point)

    return tile, 200, {
        'Content-Type': 'image/png'
    }


app.wsgi_app = DispatcherMiddleware(None, {
    app.config['APPLICATION_ROOT']: app.wsgi_app
})


if __name__ == '__main__':

    # read_viirs_tile(500, bounds=None, point='38.9072,-77.0369')

    # read_tile(
        # 'LC81920302016304LGN00',
        # # 'water',
        # 'ndvi',
        # 1000,
        # point='10.9518622,43.8459117',
        # # bounds='11.066665649414062,43.65793702655821,11.4312744140625,43.86423779837694'
    # )
    app.run(host='0.0.0.0', port=8000, debug=True)
