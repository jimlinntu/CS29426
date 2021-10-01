# Face Morphing
Name: Tzu-Chuan Lin
SID: 3036360742
Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj3/cs194-26-abu/>
Email: tzu-chuan_lin@berkeley.edu

## Environment setup
* Environment creation: `conda create -n proj3 python=3.6`
* Requirements: `pip install -r requirements.txt`

## Defining Correspondences

* `python main.py --save <the path you want to save the coorespondences>`

## Computing the "Mid-way Face" & The Morph Sequence

* Get the midway face, generate the triangulation images and the GIF
`python main.py --load ./jim_george_corres.json`

## The "Mean face" of a population

* Generate: some faces into the average shape

```
python main_part2.py ./src_imgs/jim-part2-crop2.jpg --write --shape --add_corners --load jim_parts_shape3.json
```

* Generate: my face in average shape and the average face in my shape

```
python main_part2.py ./src_imgs/jim-part2-crop2.jpg --write --shape --load jim_parts_shape3.json
```

## Caricatures: Extrapolating from the mean

* Generate: my caricatures extrapolating from the average Danish man.

```
python main_part2.py ./src_imgs/jim-part2-crop2.jpg --write --shape --add_corners --load jim_parts_shape3.json
```

## Bells and Whistles

* Change only appearance:
    * `python main_change_ethnicity.py 0.0 0.0 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.0 0.3 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.0 0.6 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.0 1.0 --load jim_change_eth.json --add_corners`
* Change only shape:
    * `python main_change_ethnicity.py 0.0 0.0 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.3 0.0 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.6 0.0 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 1.0 0.0 --load jim_change_eth.json --add_corners`
* Change both appearance and shape:
    * `python main_change_ethnicity.py 0.0 0.0 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.3 0.3 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 0.6 0.6 --load jim_change_eth.json --add_corners`
    * `python main_change_ethnicity.py 1.0 1.0 --load jim_change_eth.json --add_corners`
