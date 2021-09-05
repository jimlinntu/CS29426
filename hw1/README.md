# Project 1 - Images of the Russian Empire

Name: Tzu-Chuan Lin

SID: 3036360742

Email: tzu-chuan_lin@berkeley.edu

Web page link: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj1/cs194-26-abu/tzu-chuan_lin_proj1/index.html>

## How to Run the code
Requirements: `pip install -r requirements.txt`

* Usage:
```
main.py [--center_mask] img_path {b,g,r} {single,multiscale,mine} {ssd,ncc} {none,histeq,clahe,grey_world} result
```

For Example:

* Single scale + using g as base + SSD + grey_world + center mask: `python main.py img_path g single ssd grey_world result --center_mask`
* Multiscale + using g as base + NCC + histogram equalization + center mask: `python main.py img_path g multiscale ncc histeq result --center_mask`
* The command I use to run the result: `python main.py img_path g multiscale ssd none result --center_mask`
