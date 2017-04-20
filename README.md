## Libra Realtime Image Processor

This simple flask app generates truecolor, ndvi and false water/land images for a given Landsat8 scene.

It can also crop the image for a given bounding box or point and produce PNG images with different sizes.

The app is built to work with the libra-alexa a prototype for showing how to search for satellite imagery using voice recognition tools.

### Installation

    $ pip install -r requirements.txt

### Usage

    $ python app.py

OR

    $ docker run -p 8000:8000 developmentseed/libra-img:latest

### API

- http://localhost:8000/image/LC81870282016333LGN00?product=ndvi&resolution=2&point=lat,lon

### Credit

based on the [POSM imagery API](https://github.com/mojodna/posm-imagery-api)
