ICT Demo 2018
=================

## Description
This is a ICT demo program

## Optional arguments
`h, --help`: show this help message and exit

`--version, -v` : show program's version number and exit.


video_process.py
> input: local video file
> output: texted video and timestamp figures.

`--display DP` , `-dp DP` :  [yes | no] display option yes/no.

`--output OUTPUT`,  `-out OUTPUT`: [ yes | no ] write video out.

`--videopath PATH`,  `-path PATH`: given video path.



## Get datas


- Download models and videos from below url.

    https://drive.google.com/open?id=1Gi5_SW1qUCZ_M3pjqd0OWeKhMy1q1D3E


## Make directory under the ROOT
```
'PROJECT_ROOT/expepriments/gist/video'
```
> Put videos in this directory

```
'PROJECT_ROOT/lib/models/ssd300'
```
> put checkpoints for SSD and MLP model in this directory.




## Example of usage

```
PROJECT_PATH$ python lib/video_process.py -dp no -path 'path for videos'
```


The result will be stored at 'output' directory under the given video path.

